# FlashVSR 模型概述

> 本文档基于仓库源码整理，概述 **FlashVSR**（面向实时的扩散式流式视频超分辨率）的整体架构、核心模块、推理流程与使用方式，方便快速理解代码结构。

---

## 1. 简介

**FlashVSR** 是首个面向**实时**的、基于扩散模型的**一步（one-step）流式**视频超分辨率（Video Super-Resolution, VSR）框架。它在单张 A100 GPU 上对 768×1408 视频可达到 **约 17 FPS**，相比以往的一步扩散式 VSR 模型实现了最高 **约 12 倍** 加速，同时保持 SOTA 画质。

- 论文：[arXiv:2510.12747](https://arxiv.org/abs/2510.12747)
- 项目主页：<http://zhuang2002.github.io/FlashVSR>
- 模型权重：[FlashVSR v1](https://huggingface.co/JunhaoZhuang/FlashVSR) / [FlashVSR v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)
- 数据集：[VSR-120K](https://huggingface.co/datasets/JunhaoZhuang/VSR-120K)（120k 视频 + 180k 图像）

> ⚠️ 该项目主要为 **4× 超分** 设计与优化，建议使用 4× 设置以获得最佳效果与稳定性。

---

## 2. 核心创新点

FlashVSR 由三个互补的技术组成：

1. **三阶段蒸馏管线（Three-Stage Distillation Pipeline）**
   将多步扩散模型蒸馏为可流式处理的一步模型，实现流式超分辨率训练。

2. **局部约束稀疏注意力（Locality-Constrained Sparse Attention, LCSA）**
   在减少冗余计算的同时，弥合训练与测试分辨率之间的差距（train–test resolution gap）。这是保证高分辨率下画质的关键模块；部分第三方实现若缺失 LCSA 而退化为稠密注意力，会导致明显的画质下降。

3. **微型条件解码器（Tiny Conditional Decoder, TCDecoder）**
   在几乎不损失画质的前提下，大幅加速潜变量到像素的重建过程。

此外还构建了 **VSR-120K** 大规模数据集，支持图像与视频的联合训练。

---

## 3. 整体架构

FlashVSR 建立在 **Wan2.1** 视频扩散模型（DiT，Diffusion Transformer）之上，结合 VAE 与流式设计。数据流大致为：

```
低质量视频 (LQ)
    │
    ├──► LQ_proj_in (Buffer_LQ4x_Proj)  ── 逐 Block 注入的 LQ 条件特征
    │
    ▼
VAE Encoder（或跳过）──► 潜变量 z
    │
    ▼
WanModel (DiT) + 局部约束稀疏注意力(LCSA) + 流式 KV 缓存
    │   （一步去噪，num_inference_steps=1，DMD 蒸馏权重）
    ▼
潜变量 ẑ
    │
    ├──► Full 版：Wan2.1 VAE Decoder（高画质）
    └──► Tiny 版：TCDecoder（微型条件解码器，快速）
    │
    ▼
颜色校正（Wavelet ColorCorrector，可选）
    │
    ▼
高分辨率视频 (HR)
```

### 关键组件

| 组件 | 源码位置 | 说明 |
|------|----------|------|
| **WanModel (DiT)** | `diffsynth/models/wan_video_dit.py` | 主干扩散 Transformer，包含稀疏注意力与流式 KV 缓存 |
| **Wan2.1 VAE** | `diffsynth/models/wan_video_vae.py` | 3D 因果 VAE，Full 版用于解码（`Wan2.1_VAE.pth`） |
| **Buffer_LQ4x_Proj / Causal_LQ4x_Proj** | `examples/WanVSR/utils/utils.py` | LQ 条件投影模块，将低质量帧编码为逐层注入 DiT 的条件特征（`LQ_proj_in.ckpt`） |
| **TCDecoder** | `examples/WanVSR/utils/TCDecoder.py` | 微型条件解码器（Tiny AutoEncoder，仅解码），Tiny 版使用（`TCDecoder.ckpt`） |
| **ColorCorrector** | `diffsynth/pipelines/flashvsr_full.py` | 基于小波（wavelet）分解的颜色/亮度校正，缓解色偏 |

---

## 4. 核心模块细节

### 4.1 LQ 条件投影（Buffer_LQ4x_Proj）

- 使用 `PixelShuffle3d`（16×16 空间下采样）+ 两层 **CausalConv3d** + `RMS_norm` + `SiLU`，在时间维度做因果下采样（f → f/4）。
- 输出经过多个 `Linear` 层（`layer_num`），为 DiT 的**每一个 block** 提供独立的条件特征，在 `model_fn_wan_video` 中以 `x = x + LQ_latents[block_id]` 的形式逐层注入。
- 提供 `forward`（整段）与 `stream_forward`（流式逐 clip）两种调用方式，并维护 `conv1/conv2` 的因果缓存，支持长视频流式推理。

### 4.2 局部约束稀疏注意力（LCSA）

位于 `wan_video_dit.py`，通过 `block_sparse_attn`（MIT Han Lab 的 [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)）后端实现：

- **窗口划分**：以 `win = (2, 8, 8)` 将时空序列分块。
- **局部掩码**：`build_local_block_mask_shifted_vec_normal_slide` 生成受 `local_range` 约束的局部块掩码（推荐 9 或 11：9 更锐利，11 更稳定）。
- **Top-k 稀疏**：`generate_draft_block_mask` 先用平均池化后的 Q/K 计算 draft 注意力分数，再结合局部掩码取 **top-k**（由 `topk_ratio` 控制）保留重要块，交由 `block_sparse_attn_func` 做稀疏注意力计算。
- `local_attention_mask` 会在分辨率/宽高比变化时自动重建（修复了不同宽高比切换的伪影问题）。

> ⚠️ 该稀疏后端在 **A100 / A800（Ampere）** 上加速效果理想，在 **H200（Hopper）** 上可运行但加速有限，其它 GPU 兼容性未知。

### 4.3 流式推理与 KV 缓存

DiT 的每个 block 在流式模式（`is_stream=True`）下维护 `pre_cache_k / pre_cache_v` 跨 clip 的 KV 缓存（长度由 `kv_ratio` 控制），配合 RoPE 位置分段（`cur_process_idx`）实现**因果、可无限延伸**的流式超分，适用于长视频。

### 4.4 微型条件解码器（TCDecoder）

- 改编自 Hunyuan Video 的 Tiny AutoEncoder，**仅保留解码器**。
- 由 `MemBlock`（带时间记忆的残差块）、`TPool/TGrow`（时间池化/扩张）等组成，支持逐帧流式解码。
- 以 LQ 视频作为条件（`cond=LQ_video`），在潜变量基础上快速重建高分辨率像素，替代较重的 Wan2.1 VAE Decoder，从而加速。

### 4.5 一步去噪（DMD 蒸馏）

- 使用 `FlowMatchScheduler` 调度器，推理时 `num_inference_steps=1`、`cfg_scale=1.0`。
- 去噪权重为 `diffusion_pytorch_model_streaming_dmd.safetensors`（DMD / 分布匹配蒸馏得到的流式一步权重）。
- 文本 context 由固定 prompt 预先生成，并通过 `init_cross_kv` 初始化所有 CrossAttention 的 KV 缓存（VSR 任务本身不依赖文本提示）。

---

## 5. 推理管线（Pipelines）

仓库提供三种管线（均在 `diffsynth/pipelines/`），并有对应的推理脚本（`examples/WanVSR/`）：

| 管线类 | 解码器 | 适用场景 | v1 脚本 / v1.1 脚本 |
|--------|--------|----------|----------------------|
| `FlashVSRFullPipeline` | Wan2.1 VAE Decoder | 高画质 | `infer_flashvsr_full.py` / `infer_flashvsr_v1.1_full.py` |
| `FlashVSRTinyPipeline` | TCDecoder | 快速 | `infer_flashvsr_tiny.py` / `infer_flashvsr_v1.1_tiny.py` |
| `FlashVSRTinyLongPipeline` | TCDecoder（流式） | 长视频 | `infer_flashvsr_tiny_long_video.py` / `infer_flashvsr_v1.1_tiny_long_video.py` |

### 主要推理参数（`pipe(...)`）

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `LQ_video` | 低质量输入视频张量 `(1, C, F, H, W)`，范围 `[-1, 1]` | — |
| `num_inference_steps` | 去噪步数（一步） | `1` |
| `cfg_scale` | 无分类器引导强度 | `1.0` |
| `topk_ratio` | 稀疏注意力 top-k 比例（越大越稳定但更慢） | `1.5` 或 `2.0` |
| `kv_ratio` | 流式 KV 缓存长度比例 | `3.0` |
| `local_range` | 局部注意力范围 | `9`（更锐利）/ `11`（更稳定） |
| `tiled` | 是否分块解码（省显存但更慢） | `False`（显存充足时）|
| `color_fix` | 是否启用小波颜色校正 | `True` |

---

## 6. 输入预处理约定

推理脚本（如 `infer_flashvsr_full.py`）对输入有若干约束：

- **分辨率**：先按 `scale`（默认 4）放大，再中心裁剪为 **128 的整数倍**（`compute_scaled_and_target_dims`）。
- **帧数**：末尾补 4 帧后，取满足 **8n+1** 的最大帧数（`largest_8n1_leq`），以适配 VAE 的时间下采样。
- **像素范围**：转换到 `[-1, 1]`，`dtype=bfloat16`。
- 支持输入为**视频文件**（mp4/mov/avi/mkv）或**图像序列文件夹**。

---

## 7. 快速上手

```bash
# 1. 克隆并安装
git clone https://github.com/OpenImagingLab/FlashVSR
cd FlashVSR
conda create -n flashvsr python=3.11.13 && conda activate flashvsr
pip install -e .
pip install -r requirements.txt

# 2. 安装 Block-Sparse-Attention（LCSA 后端，必需）
git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention && pip install packaging ninja && python setup.py install

# 3. 下载权重（Git LFS）到 examples/WanVSR/
cd examples/WanVSR
git lfs install
git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1   # 推荐 v1.1

# 4. 运行推理
python infer_flashvsr_v1.1_full.py
```

下载后的权重目录结构：

```
FlashVSR(-v1.1)/
├── LQ_proj_in.ckpt                                      # LQ 条件投影权重
├── TCDecoder.ckpt                                       # 微型解码器权重（Tiny 版）
├── Wan2.1_VAE.pth                                       # VAE 权重（Full 版解码）
├── diffusion_pytorch_model_streaming_dmd.safetensors   # 一步流式 DiT（DMD 蒸馏）
└── README.md
```

---

## 8. 目录结构速览

```
FlashVSR/
├── diffsynth/                     # 核心库（基于 DiffSynth-Studio 改造）
│   ├── models/
│   │   ├── wan_video_dit.py       # DiT 主干 + LCSA 稀疏注意力 + 流式 KV
│   │   └── wan_video_vae.py       # Wan2.1 3D 因果 VAE
│   ├── pipelines/
│   │   ├── flashvsr_full.py       # Full 管线（VAE 解码）
│   │   ├── flashvsr_tiny.py       # Tiny 管线（TCDecoder 解码）
│   │   └── flashvsr_tiny_long.py  # Tiny 长视频流式管线
│   └── schedulers/flow_match.py   # FlowMatch 调度器
├── examples/WanVSR/
│   ├── infer_flashvsr_*.py        # 各版本推理脚本
│   └── utils/
│       ├── utils.py               # Buffer/Causal_LQ4x_Proj 等
│       └── TCDecoder.py           # 微型条件解码器
├── requirements.txt
└── setup.py
```

---

## 9. 参考与致谢

- **论文引用**

  ```bibtex
  @article{zhuang2025flashvsr,
    title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution},
    author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
    journal={arXiv preprint arXiv:2510.12747},
    year={2025}
  }
  ```

- 依赖的开源项目：
  - [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
  - [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
  - [taehv](https://github.com/madebyollin/taehv)

> 更完整的说明、社区集成（ComfyUI 等）与画质对比请参见仓库 [`README.md`](./README.md)。
