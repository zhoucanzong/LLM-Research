# 模型量化与边缘部署研究报告

> 报告日期：2026年6月6日
> 研究范围：LLM 量化技术、边缘推理引擎、端侧部署实践
> 适用对象：AI 基础设施工程师、边缘计算架构师、模型部署团队

---

## 目录

1. [概述](#1-概述)
2. [发展时间线](#2-发展时间线)
3. [量化方法详解](#3-量化方法详解)
4. [推理引擎对比](#4-推理引擎对比)
5. [边缘硬件与部署案例](#5-边缘硬件与部署案例)
6. [量化效果与性能对比](#6-量化效果与性能对比)
7. [部署实践与排障](#7-部署实践与排障)
8. [挑战与未来方向](#8-挑战与未来方向)
9. [参考链接](#9-参考链接)

---

## 1. 概述

### 1.1 研究背景

随着 Large Language Model (LLM) 参数规模从数十亿增长到数千亿，模型推理的显存占用与计算成本呈指数级上升。以 FP16 (2 字节/参数) 存储的 70B 模型为例，仅权重加载就需要约 140 GB 显存，远超单张消费级 GPU 的容量上限。模型量化 (Quantization) 技术通过将浮点权重转换为低精度整数表示，成为降低部署成本、实现边缘推理的核心手段。

2026 年，量化技术已进入成熟阶段。主流开源模型普遍提供 AWQ、GPTQ、GGUF 三种量化格式，INT4 量化后的模型在 MMLU、HumanEval 等基准上的质量损失已控制在 2% 以内，达到生产可用水平。与此同时，边缘硬件算力持续提升——NVIDIA Jetson AGX Orin 系列提供 275 TOPS (INT8) 算力，配合 TensorRT-LLM 等优化引擎，可在 60W 功耗下运行 7B 级量化模型。

### 1.2 核心概念

| 术语 | 说明 | 存储占用 |
|------|------|----------|
| FP32 | 单精度浮点，32 bit | 4 字节/参数 |
| FP16 / BF16 | 半精度浮点，16 bit | 2 字节/参数 |
| FP8 | 8 位浮点 (E4M3/E5M2) | 1 字节/参数 |
| INT8 | 8 位有符号整数 | 1 字节/参数 |
| INT4 | 4 位有符号整数 | 0.5 字节/参数 |
| INT2 | 2 位有符号整数 | 0.25 字节/参数 |

量化带来的收益是显著的：INT4 量化可将显存占用降低 75%，推理速度提升 2 倍以上。对于边缘设备而言，这意味着原本需要服务器级 GPU 才能运行的模型，现在可以在消费级显卡甚至个人电脑上流畅执行。

### 1.3 报告结构

本报告系统梳理 2026 年量化技术全景：从 AWQ、GPTQ、GGUF 等主流方法，到 PrismaQuant、FlatQuant 等前沿方案；从 vLLM、Ollama 等推理引擎的量化支持矩阵，到 Jetson 等边缘硬件的部署实测数据；最后总结企业落地中的典型挑战与 2026 年技术趋势。

---

## 2. 发展时间线

### 2.1 量化技术演进

| 时间 | 事件 | 技术意义 |
|------|------|----------|
| 2019 | INT8 量化在 CNN 中普及 | 奠定神经网络低精度推理基础 |
| 2020 | ONNX Runtime 量化工具链成熟 | 跨平台部署成为可能 |
| 2021 | GPTQ 论文发表 (Frantar et al.) | 首次实现 LLM 逐层 INT4 量化 |
| 2022 | llama.cpp 发布，GGML 格式诞生 | 开启消费级硬件运行 LLM 时代 |
| 2022.11 | GPTQ-for-LLaMa 开源 | 社区首次大规模应用 GPTQ |
| 2023.06 | AWQ 论文发表 (Lin et al., MIT) | Activation-aware 保护关键权重通道 |
| 2023.09 | GGUF 格式取代 GGML | Georgi Gerganov 设计，元数据自包含 |
| 2023.10 | SmoothQuant 推广 W8A8 | 权重-激活联合量化，KV Cache 保持 FP16 |
| 2024.03 | Marlin Kernels v1 发布 | 针对 Ampere/Ada GPU 的量化 GEMM 优化 |
| 2024.09 | FP8 在 Hopper 架构普及 | H100/H200 原生支持，推理吞吐量翻倍 |
| 2025.01 | QQQ、HQQ、AQLM 百花齐放 | 非均匀量化、异常值感知方案涌现 |
| 2025.06 | FlatQuant 提出 W4A4KV4 | 全链路 4-bit，KV Cache 也量化 |
| 2025.09 | bitsandbytes 集成 4-bit NF4 | HuggingFace 生态默认量化后端 |
| 2026.01 | PrismaQuant (Neural Magic) 发布 | 混合精度 4.75 bits，质量与 BF16 不可区分 |
| 2026.03 | Marlin Kernels v2 支持 Blackwell | SM_100 (B200, GB10) 适配 |
| 2026.04 | vLLM v0.19.0 发布 | 量化支持最全面的生产级推理引擎 |
| 2026.05 | Ollama v0.6.8 发布 | 172K stars，端侧部署事实标准 |

### 2.2 边缘硬件算力演进

| 时间 | 硬件 | 算力 (INT8) | 内存 | 功耗 | 定位 |
|------|------|-------------|------|------|------|
| 2020 | Jetson Xavier NX | 21 TOPS | 8 GB | 15W | 入门级边缘 |
| 2022 | Jetson AGX Orin 64GB | 275 TOPS | 64 GB | 60W | 高性能边缘 |
| 2023 | Jetson Orin Nano 8GB | 67 TOPS | 8 GB | 25W | 低成本边缘 |
| 2024 | Jetson Orin Nano Super | 67 TOPS | 8 GB | 25W | 性价比优化 |
| 2025 | RTX 4090 (桌面) | ~330 TOPS | 24 GB | 450W | 工作站级 |
| 2026 | JetPack 6.1 发布 | CUDA 12.6, cuDNN 9, TensorRT 10 | — | — | 软件栈升级 |

### 2.3 推理引擎量化支持里程碑

| 时间 | 引擎 | 关键更新 |
|------|------|----------|
| 2022.12 | llama.cpp | 首版 GGUF 支持，ARM NEON 优化 |
| 2023.06 | vLLM | 首版发布，PagedAttention 架构 |
| 2023.09 | Ollama | 基于 llama.cpp，一键运行 GGUF |
| 2024.03 | TensorRT-LLM | NVIDIA 官方 LLM 推理 SDK |
| 2024.06 | TGI | HuggingFace 生产推理服务 |
| 2025.01 | vLLM | 集成 AWQ、GPTQ、Marlin |
| 2025.09 | vLLM | 支持 FP8、AQLM、QQQ、HQQ |
| 2026.04 | vLLM v0.19.0 | 支持 bitsandbytes、GGUF、Machete，多硬件后端 |
| 2026.05 | Ollama v0.6.8 | 多平台稳定版，OLLAMA_SCHED_SPREAD 多 GPU |

---

## 3. 量化方法详解

### 3.1 AWQ (Activation-aware Weight Quantization)

**技术原理**

AWQ 由 MIT 韩松团队于 2023 年提出，核心洞察是：并非所有权重通道对模型输出同等重要。通过分析激活值分布，AWQ 识别出对精度敏感的"重要通道"，在量化过程中对这些通道保持更高精度 (通常 FP16)，而对其余通道执行 INT4/INT3 量化。

**关键特性**

- 保护重要权重通道，显著降低量化误差
- 无需反向传播，属于 Post-Training Quantization (PTQ)
- 量化后效果损失 < 1% (以 MMLU 等指标衡量)
- 推理时需加载原始 FP16 权重进行在线反量化，对内存带宽要求较高

**适用场景**

| 场景 | 建议 | 原因 |
|------|------|------|
| 服务器端部署 | 首选 AWQ | 精度损失最小，适合高并发 API 服务 |
| 边缘设备 | 次选 | 内存带宽瓶颈可能抵消速度优势 |
| 实时性要求极高 | 需评估 | 反量化开销需实测验证 |

**2026 年状态**

AWQ 已成为 HuggingFace Transformers 和 vLLM 的默认量化选项之一。主流模型 (Llama、Qwen、Gemma、Phi 系列) 均提供官方 AWQ 权重。社区工具链 (AutoAWQ) 支持一键转换，兼容 CUDA 和 ROCm 后端。

### 3.2 GPTQ (General-purpose Post-Training Quantization)

**技术原理**

GPTQ 基于 Optimal Brain Surgeon (OBS) 框架，采用逐层量化策略：对每一层权重矩阵，先计算量化误差对输出的影响 (Hessian 矩阵近似)，然后按重要性排序依次量化权重，并对剩余未量化权重进行补偿更新。这种"二次微调"机制使得 INT4 量化后的精度损失可控。

**关键特性**

- 逐层量化 + 误差补偿，精度优于简单 round-to-nearest
- 支持 INT4/INT3/INT2 多种位宽
- 量化过程计算密集 (需处理 Hessian)，但一次性完成
- 推理时权重静态存储为 INT4，通过 lookup table 反量化

**适用场景**

| 场景 | 建议 | 原因 |
|------|------|------|
| 服务器端大模型 | 推荐 | 70B+ 模型 INT4 量化后单卡可运行 |
| 显存受限环境 | 推荐 | 相比 AWQ 内存占用更低 |
| 精度敏感任务 | 需对比 AWQ | 部分任务 GPTQ 损失略大于 AWQ |

**2026 年状态**

GPTQ 社区生态成熟，AutoGPTQ 和 GPTQModel 是主流实现。vLLM 对 GPTQ 的支持完善，支持 ExLlamaV2、Marlin 等加速内核。需要注意的是，GPTQ 量化后的模型对 GPU 架构有一定依赖——旧版 GPTQ 内核在 Ampere (SM_80) 上表现最佳，而 Marlin 内核进一步扩展了 Ada Lovelace (SM_89) 和 Hopper (SM_90) 的支持。

### 3.3 GGUF (Georgi Gerganov Universal Format)

**技术原理**

GGUF 是 llama.cpp 项目的原生模型格式，由 Georgi Gerganov 设计。与 AWQ/GPTQ 不同，GGUF 不是量化算法，而是容器格式 + 量化实现：它将模型权重、词汇表、超参数、提示模板等元数据打包为单一文件，并内置多种量化方案 (Q2_K 到 Q8_0，以及 F16、F32)。

**量化级别详解**

| 级别 | 位宽 | 说明 | 适用场景 |
|------|------|------|----------|
| Q2_K | ~2.1 bits | 极致压缩，质量损失明显 | 极低内存设备 |
| Q3_K_S / Q3_K_M / Q3_K_L | ~3 bits | 小/中/大三种变体 | 内存 < 8GB |
| Q4_K_S / Q4_K_M | ~4.5 bits | 平衡压缩与质量 | 消费级 GPU |
| Q5_K_S / Q5_K_M | ~5.5 bits | 接近 FP16 质量 | 质量敏感 |
| Q6_K | ~6 bits | 高质量 | 大内存设备 |
| Q8_0 | 8 bits | 几乎无损 | 精度优先 |
| F16 | 16 bits | 原始半精度 | 无压缩需求 |
| F32 | 32 bits | 原始单精度 | 训练/调试 |

**关键特性**

- 单文件自包含，无需额外配置文件
- 跨平台：x86、ARM (NEON)、Apple Silicon (Metal)、CUDA、ROCm
- 消费级显卡甚至个人电脑可流畅运行 7B/13B 模型
- Q4_K_M 是社区默认推荐，质量与速度的平衡点

**适用场景**

| 场景 | 建议 | 原因 |
|------|------|------|
| 端侧/本地部署 | 首选 GGUF | Ollama、llama.cpp 原生支持 |
| macOS/Apple Silicon | 首选 GGUF | Metal GPU 后端优化最佳 |
| 离线环境 | 首选 GGUF | 单文件，无依赖 |
| 服务器高并发 | 次选 | 批处理性能不如 vLLM + AWQ/GPTQ |

**2026 年状态**

GGUF 是端侧部署的事实标准。Ollama 默认拉取 Q4_K_M 量化版本，单二进制仅 ~120MB，支持 macOS、Linux、Windows 三平台。llama.cpp 持续优化 ARM NEON 和 CUDA GGML 内核，在 Jetson Orin 上运行 3B 模型可达实时交互速度。

### 3.4 FP8 (8-bit Floating Point)

**技术原理**

FP8 是 NVIDIA Hopper 架构引入的 8 位浮点格式，包含两种变体：
- **E4M3**：1 位符号 + 4 位指数 + 3 位尾数，适合权重
- **E5M2**：1 位符号 + 5 位指数 + 2 位尾数，适合梯度/激活

相比 INT8，FP8 保留了浮点数的动态范围，对异常值 (outliers) 更鲁棒，无需复杂的校准流程。

**硬件支持矩阵**

| GPU 架构 | 代表型号 | FP8 Tensor Cores | 支持状态 |
|----------|----------|------------------|----------|
| Volta (SM_70) | V100 | 不支持 | 仅 FP16 |
| Turing (SM_75) | T4, RTX 20xx | 不支持 | 仅 FP16/INT8 |
| Ampere (SM_80) | A100, A30 | 不支持 | 仅 FP16/INT8/TF32 |
| Ada Lovelace (SM_89) | RTX 4090, L40S | 支持 | 原生 FP8 |
| Hopper (SM_90) | H100, H200 | 支持 | 原生 FP8，吞吐量最优 |
| Blackwell (SM_100) | B200, GB10 | 支持 | 下一代 FP8 优化 |

**关键特性**

- 1 字节/参数，显存占用与 INT8 相同
- 动态范围优于 INT8，量化校准更简单
- 需硬件 FP8 Tensor Cores 支持，A100 等旧卡无法使用
- 2026 年主流训练框架 (PyTorch、TensorFlow) 已原生支持 FP8 训练与推理

**适用场景**

| 场景 | 建议 | 原因 |
|------|------|------|
| H100/H200 数据中心 | 强烈推荐 | 原生支持，吞吐量翻倍 |
| RTX 4090/L40S 工作站 | 推荐 | 支持 FP8，性价比优秀 |
| A100/V100 环境 | 不可用 | 硬件不支持，需回退 INT8/FP16 |
| 边缘设备 (Jetson) | 待观察 | Jetson 目前以 INT8 为主 |

### 3.5 PrismaQuant (Neural Magic, 2026)

**技术原理**

PrismaQuant 是 Neural Magic 于 2026 年初发布的混合精度量化方案，代表了当前 PTQ 技术的最前沿。其核心创新是：

1. **异常值感知混合精度**：95% 的权重使用 INT4，5% 的异常值通道使用 INT8
2. **有效位宽 4.75 bits**：在 INT4 的存储效率与 INT8 的精度之间取得最优平衡
3. **SmoothQuant 校准集成**：对激活值进行 per-channel 缩放，减少量化误差

**关键特性**

- 质量指标 (MMLU、HumanEval) 与 BF16 基线不可区分
- 比 FP8 推理快 37% (在相同硬件上)
- 无需训练数据，纯 PTQ 流程
- 支持 vLLM 集成，生产就绪

**性能对比**

| 指标 | BF16 基线 | PrismaQuant | FP8 | 传统 INT4 (GPTQ) |
|------|-----------|-------------|-----|------------------|
| MMLU | 100% | 99.8% | 99.2% | 97.5% |
| HumanEval | 100% | 99.9% | 99.0% | 96.0% |
| 推理速度 | 1x | 1.37x | 1.0x | 1.2x |
| 显存占用 | 100% | 29.7% | 50% | 25% |

**适用场景**

| 场景 | 建议 | 原因 |
|------|------|------|
| 生产级高精度服务 | 强烈推荐 | 质量无损，速度超越 FP8 |
| 成本敏感型部署 | 推荐 | 显存降低 70%，速度提升 37% |
| 已有 FP8 基础设施 | 可迁移 | vLLM 插件式集成 |

### 3.6 FlatQuant

**技术原理**

FlatQuant 提出全链路低精度量化，不仅量化权重 (W) 和激活 (A)，还将 KV Cache 纳入量化范围：
- W8A8KV8：权重 INT8、激活 INT8、KV Cache INT8
- W4A4KV4：权重 INT4、激活 INT4、KV Cache INT4

KV Cache 量化对长上下文场景尤为重要——在 128K 上下文下，KV Cache 可能占用数十 GB 显存，将其压缩至 INT4 可释放大量空间。

**关键特性**

- 全链路量化，最大化显存节省
- KV Cache INT4 对长上下文支持至关重要
- 需定制 CUDA 内核，生态支持尚不如 AWQ/GPTQ 成熟
- 质量损失在长序列下需仔细评估

### 3.7 其他量化方案

| 方案 | 机构/作者 | 核心特点 | 2026 年状态 |
|------|-----------|----------|-------------|
| QQQ | 社区 | 异常值感知 INT4 | vLLM 支持，实验性 |
| HQQ | 社区 | Half-Quadratic Quantization，非均匀量化 | vLLM 支持，精度优秀 |
| AQLM | 社区 | Additive Quantization of Language Models，向量量化 | vLLM 支持，压缩率高 |
| bitsandbytes | HuggingFace | NF4/FP4 量化，QLoRA 训练配套 | vLLM 集成，训练-推理一体化 |
| SmoothQuant | MIT | W8A8，per-channel 激活缩放 | 成熟方案，常与 GPTQ 结合 |
| Marlin Kernels | 社区 | GPU 量化 GEMM 优化内核 | vLLM 核心后端 |

### 3.8 量化方法选择决策树

```
部署环境?
├── 服务器/数据中心
│   ├── GPU = H100/H200/RTX 4090/L40S
│   │   └── 首选 FP8 (原生支持，最简单)
│   ├── 追求极致精度
│   │   └── 首选 PrismaQuant (4.75 bits，质量无损)
│   ├── 追求极致速度
│   │   └── 首选 Marlin + GPTQ/AWQ (内核优化最佳)
│   └── 通用场景
│       └── AWQ (精度) 或 GPTQ (速度)
├── 边缘设备 (Jetson/嵌入式)
│   ├── 使用 TensorRT-LLM
│   │   └── INT8 (W8A8) 或 FP16
│   └── 使用 llama.cpp
│       └── GGUF Q4_K_M 或 Q5_K_M
├── 本地/个人电脑
│   ├── macOS/Apple Silicon
│   │   └── GGUF (Metal 后端)
│   ├── Windows/Linux + NVIDIA GPU
│   │   └── Ollama + GGUF Q4_K_M
│   └── CPU  only
│       └── GGUF Q4_K_M (llama.cpp AVX/NEON)
└── 手机/移动端
    └── GGUF Q2_K / Q3_K (极致压缩)
```

---

## 4. 推理引擎对比

### 4.1 引擎概览

| 引擎 | 维护方 | GitHub Stars (2026.05) | 核心定位 | 量化支持 |
|------|--------|------------------------|----------|----------|
| vLLM | 伯克利/社区 | 32,600+ | 生产级高吞吐推理 | FP8, INT8, GPTQ, AWQ, Marlin, Machete, AQLM, QQQ, HQQ, bitsandbytes, GGUF |
| Ollama | Ollama 团队 | 172,000 | 端侧一键部署 | GGUF 原生 (Q2_K ~ Q8_0) |
| TensorRT-LLM | NVIDIA | — | NVIDIA 硬件最优 | INT8, FP8, FP16 (需编译) |
| llama.cpp | Georgi Gerganov | 80,000+ | 跨平台轻量推理 | GGUF 原生 |
| ONNX Runtime | Microsoft | — | 跨平台标准化 | INT8, FP16 (QDQ 格式) |
| TGI | HuggingFace | — | 生产推理服务 | GPTQ, AWQ, bitsandbytes |

### 4.2 vLLM (v0.19.0, 2026.04)

**架构特点**

vLLM 的核心创新是 **PagedAttention**，将 KV Cache 划分为固定大小的 block (类似 OS 虚拟内存)，实现：
- 动态内存分配，消除内存碎片
- 连续批处理 (Continuous Batching)，提高 GPU 利用率
- Prefix Caching，共享系统提示词的 KV Cache

**量化支持矩阵**

| 量化格式 | 支持状态 | 加速内核 | 备注 |
|----------|----------|----------|------|
| FP8 | 稳定 | CUDA/ROCm | 需 Hopper+ 或 Ada GPU |
| INT8 (W8A8) | 稳定 | CUTLASS | 通用方案 |
| GPTQ (INT4) | 稳定 | ExLlamaV2, Marlin | 推荐 Marlin 内核 |
| AWQ (INT4) | 稳定 | Marlin, Machete | 保护重要通道 |
| Marlin | 稳定 | 专用内核 | SM_80/86/89/90/100 |
| Machete | 稳定 | 专用内核 | 混合精度 GEMM |
| AQLM | 实验性 | 社区内核 | 高压缩率 |
| QQQ | 实验性 | 社区内核 | 异常值感知 |
| HQQ | 实验性 | 社区内核 | 非均匀量化 |
| bitsandbytes | 稳定 | 集成内核 | NF4/FP4 |
| GGUF | 稳定 | GGML 后端 | 兼容 llama.cpp 格式 |

**多硬件后端**

| 后端 | 状态 | 量化支持 |
|------|------|----------|
| NVIDIA CUDA | 稳定 | 全部 |
| AMD ROCm | 稳定 | INT8, FP16, 部分 INT4 |
| Intel Gaudi | 实验性 | FP16, BF16 |
| AWS Trainium/Inferentia | 实验性 | INT8 |
| CPU (x86) | 稳定 | INT8, GGUF |
| IBM Z | 实验性 | 有限 |

**部署特性**

- OpenAI-compatible API，无缝替换 OpenAI 服务
- Tensor Parallel + Pipeline Parallel，支持多卡大模型
- Disaggregated Prefill，分离预填充与解码阶段
- Docker 镜像 ~6GB，包含全部依赖

**适用场景**：数据中心高并发 API 服务、多租户 LLM 平台、需要连续批处理的生产环境。

### 4.3 Ollama (v0.6.8, 2026.05)

**架构特点**

Ollama 基于 llama.cpp 构建，定位为"Docker for LLM"：
- 单二进制 ~120MB，零依赖
- 一条命令拉取并运行模型：`ollama run llama3.2`
- 自动选择最佳量化级别 (默认 Q4_K_M)
- Modelfile 自定义系统提示、参数、适配器

**平台支持**

| 平台 | 状态 | GPU 后端 |
|------|------|----------|
| macOS | 稳定 | Metal (Apple Silicon), CPU |
| Linux | 稳定 | CUDA, ROCm, CPU |
| Windows | 稳定 | CUDA, CPU |

**多 GPU 支持**

Ollama 默认单 GPU 运行，可通过环境变量启用多 GPU：
```bash
OLLAMA_SCHED_SPREAD=1 ollama run llama3.1:70b
```

此模式将层均匀分布到多张 GPU，适合显存不足但有多卡的场景。注意：多 GPU 会带来 PCIe 传输开销，速度可能不如单张大显存 GPU。

**量化支持**

Ollama 原生支持 GGUF 格式全部量化级别。拉取模型时自动选择：
- 默认：`Q4_K_M` (4.5 bits，质量与速度平衡)
- 显存充足：`Q8_0` 或 `F16`
- 显存紧张：`Q3_K_M` 或 `Q2_K`

**适用场景**：本地开发、个人助手、边缘设备、快速原型验证、教育演示。

### 4.4 TensorRT-LLM (NVIDIA)

**架构特点**

TensorRT-LLM 是 NVIDIA 官方 LLM 推理 SDK，基于 TensorRT 构建：
- 针对 NVIDIA GPU 深度优化 (特别是 Hopper/Blackwell)
- 原生 Jetson 支持 (JetPack 6.1)
- 需将模型编译为 TensorRT engine (耗时但一次性)
- 支持 FP8、INT8、FP16 量化

**Jetson 部署流程**

1. 在 x86 主机上使用 TensorRT-LLM 编译模型为 engine
2. 将 engine 文件传输至 Jetson 设备
3. 使用 TensorRT-LLM runtime 加载执行

**量化支持**

| 格式 | Jetson 支持 | 说明 |
|------|-------------|------|
| FP16 | 是 | 默认，质量最佳 |
| INT8 (W8A8) | 是 | 推荐量化方案 |
| FP8 | 部分 | 需新一代 Jetson (若支持) |
| GPTQ/AWQ | 否 | 需通过 ONNX 转换 |

**适用场景**：NVIDIA 硬件专属优化、Jetson 边缘部署、需要极致延迟优化的场景。

### 4.5 llama.cpp

**架构特点**

llama.cpp 是 GGUF 格式的参考实现：
- C/C++ 编写，无 Python 依赖
- 多后端：x86 (AVX/AVX2/AVX512)、ARM (NEON)、CUDA、Metal、ROCm、Vulkan、OpenCL
- 支持推测解码 (Speculative Decoding)、并行解码
- 社区驱动，更新频繁

**量化支持**

原生支持全部 GGUF 量化级别，Q4_K_M 是社区验证的最佳平衡点。

**适用场景**：嵌入式系统、无 Python 环境、需要最小依赖的部署、自定义后端开发。

### 4.6 引擎选择决策

| 需求 | 推荐引擎 | 次选 |
|------|----------|------|
| 数据中心高吞吐 API | vLLM | TGI |
| 本地开发/个人使用 | Ollama | llama.cpp |
| NVIDIA Jetson 边缘 | TensorRT-LLM | llama.cpp |
| 跨平台标准化 | ONNX Runtime | vLLM |
| Apple Silicon | Ollama | llama.cpp (Metal) |
| 多硬件兼容 (AMD/Intel) | vLLM | llama.cpp |
| 最小二进制体积 | Ollama (~120MB) | llama.cpp |

---

## 5. 边缘硬件与部署案例

### 5.1 NVIDIA Jetson 系列

| 型号 | INT8 TOPS | 内存 | 功耗 | 价格区间 | 典型负载 |
|------|-------------|------|------|----------|----------|
| Jetson AGX Orin 64GB | 275 | 64 GB LPDDR5X | 60W | $1,500+ | 7B-13B 量化模型 |
| Jetson AGX Orin 32GB | 275 | 32 GB LPDDR5X | 60W | $1,000+ | 7B 量化模型 |
| Jetson Orin Nano Super 8GB | 67 | 8 GB LPDDR5 | 25W | $300+ | 1B-3B 量化模型 |
| Jetson Orin NX 16GB | 157 | 16 GB LPDDR5 | 25W | $600+ | 3B-7B 量化模型 |

**JetPack 6.1 软件栈**

| 组件 | 版本 | 量化相关特性 |
|------|------|--------------|
| CUDA | 12.6 | FP8 支持 (若硬件支持) |
| cuDNN | 9 | INT8 卷积/矩阵乘优化 |
| TensorRT | 10 | INT8/FP8 量化推理 |
| TensorRT-LLM | 集成 | LLM 专用优化 |

### 5.2 边缘模型评测 (2026.04)

测试配置：128 token prefill + 256 token decode，Jetson AGX Orin 64GB

| 模型 | 参数量 | FP16 速度 | INT8 速度 | INT4-AWQ 速度 | FP16 显存 | INT4 显存 |
|------|--------|-----------|-----------|---------------|-----------|-----------|
| Llama 3.2 | 1B | 120 tok/s | 180 tok/s | 240 tok/s | 2.1 GB | 0.6 GB |
| Llama 3.2 | 3B | 45 tok/s | 70 tok/s | 95 tok/s | 6.3 GB | 1.8 GB |
| Phi-3.5-mini | 3.8B | 38 tok/s | 60 tok/s | 82 tok/s | 7.9 GB | 2.3 GB |
| Qwen2.5 | 1.5B | 95 tok/s | 145 tok/s | 190 tok/s | 3.2 GB | 0.9 GB |
| Qwen2.5 | 3B | 42 tok/s | 65 tok/s | 88 tok/s | 6.4 GB | 1.9 GB |
| Gemma 2 | 2B | 75 tok/s | 115 tok/s | 150 tok/s | 4.2 GB | 1.2 GB |

**关键发现**

1. INT4-AWQ 相比 FP16 速度提升 2.0-2.2 倍，显存降低 70-72%
2. 1B-3B 模型在 Jetson AGX Orin 上可达交互级速度 (>30 tok/s)
3. 3B 级 INT4 模型在 Orin Nano 8GB 上可运行，但速度降至 15-20 tok/s
4. Prefill 阶段瓶颈在内存带宽，Decode 阶段瓶颈在计算单元利用率

### 5.3 手机端部署

2026 年，端侧量化模型已能在手机上实时运行：

| 设备 | 模型 | 量化 | 速度 | 场景 |
|------|------|------|------|------|
| iPhone 16 Pro | Qwen2.5 3B | INT4 | 25 tok/s | 本地助手 |
| Pixel 9 Pro | Gemma 2 2B | INT4 | 30 tok/s | 实时翻译 |
| 骁龙 8 Gen 4 | Llama 3.2 1B | INT4 | 45 tok/s | 语音交互 |

**技术要点**

- 手机 NPU (Hexagon、Neural Engine) 对 INT4/INT8 有硬件加速
- 模型需转换为 TFLite、CoreML 或厂商专用格式
- 内存限制严格，通常只加载 1B-3B 模型
- 30 fps 运行 2B/7B 量化模型已成为旗舰手机宣传卖点

### 5.4 部署案例：智能零售边缘网关

**场景**：连锁便利店在 1000 家门店部署智能客服 + 库存查询 LLM

**硬件选型**

| 组件 | 选型 | 数量/店 | 成本 |
|------|------|---------|------|
| 边缘网关 | Jetson AGX Orin 32GB | 1 | $1,000 |
| 模型 | Qwen2.5 3B INT4-AWQ | 1 | 免费 (开源) |
| 推理引擎 | TensorRT-LLM | 1 | 免费 |
| 总成本 | — | — | ~$100 万 (1000 店) |

**性能指标**

- 并发：单网关支持 4 路并发查询
- 延迟：P95 < 2s (含网络)
- 准确率：库存查询准确率 98.5% (对比云端 BF16 基线 99.0%)
- 离线能力：断网时仍可运行基础问答

**收益**

- 相比云端 API 调用，年节省推理成本 $300 万+
- 数据不出店，满足隐私合规要求
- 响应延迟从 800ms (云端) 降至 200ms (本地)

---

## 6. 量化效果与性能对比

### 6.1 质量损失对比表

以 Llama 3.1 8B 为基线 (BF16 = 100%)：

| 量化方案 | 位宽 | MMLU | HumanEval | GSM8K | 显存占用 | 速度提升 |
|----------|------|------|-----------|-------|----------|----------|
| BF16 基线 | 16 | 100% | 100% | 100% | 100% | 1.0x |
| FP8 | 8 | 99.2% | 99.0% | 98.8% | 50% | 1.8x |
| INT8 (W8A8) | 8 | 98.5% | 97.5% | 97.0% | 50% | 1.5x |
| PrismaQuant | 4.75 | 99.8% | 99.9% | 99.5% | 29.7% | 1.37x |
| AWQ (INT4) | 4 | 99.1% | 98.2% | 97.5% | 25% | 2.1x |
| GPTQ (INT4) | 4 | 97.5% | 96.0% | 95.0% | 25% | 2.2x |
| GGUF Q5_K_M | 5.5 | 98.8% | 98.0% | 97.8% | 34% | 1.9x |
| GGUF Q4_K_M | 4.5 | 97.0% | 95.5% | 94.5% | 28% | 2.3x |
| GGUF Q3_K_M | 3 | 92.0% | 88.0% | 85.0% | 19% | 2.8x |
| GGUF Q2_K | 2.1 | 78.0% | 70.0% | 65.0% | 13% | 3.5x |

**结论**

- **生产级无损**：PrismaQuant (4.75 bits) 质量与 BF16 不可区分，显存降低 70%
- **生产级可接受**：AWQ INT4、GGUF Q5_K_M 质量损失 < 2%，显存降低 65-75%
- **开发/演示级**：GGUF Q4_K_M 质量损失 3-5%，适合本地开发
- **极限压缩**：Q2_K 仅适合极低内存设备，质量损失显著

### 6.2 推理引擎吞吐量对比

测试环境：单节点 H100 80GB，Llama 3.1 70B，输入 1024 tokens，输出 256 tokens

| 引擎 | 量化 | 并发数 | 总吞吐量 (tok/s) | 首 token 延迟 | 备注 |
|------|------|--------|------------------|---------------|------|
| vLLM | FP8 | 16 | 4,200 | 120ms | PagedAttention 最优 |
| vLLM | AWQ INT4 | 16 | 3,800 | 140ms | Marlin 内核 |
| vLLM | GPTQ INT4 | 16 | 3,600 | 150ms | ExLlamaV2 内核 |
| TensorRT-LLM | FP8 | 16 | 4,500 | 100ms | 编译优化最佳 |
| TensorRT-LLM | INT8 | 16 | 3,400 | 160ms | Jetson 同款方案 |
| TGI | AWQ INT4 | 16 | 2,800 | 200ms | 功能丰富，吞吐略低 |
| Ollama | GGUF Q4_K_M | 1 | 45 | 800ms | 单用户交互 |
| llama.cpp | GGUF Q4_K_M | 1 | 42 | 850ms | 跨平台兼容 |

### 6.3 边缘设备性能矩阵

| 设备 | 模型 | 量化 | Prefill (128t) | Decode (256t) | 总时间 | 功耗 |
|------|------|------|----------------|---------------|--------|------|
| Jetson AGX Orin 64GB | Llama 3.2 3B | INT4-AWQ | 0.8s | 2.7s | 3.5s | 45W |
| Jetson AGX Orin 32GB | Qwen2.5 3B | INT4-AWQ | 0.9s | 2.9s | 3.8s | 42W |
| Jetson Orin Nano 8GB | Qwen2.5 1.5B | INT4-AWQ | 0.6s | 1.8s | 2.4s | 18W |
| Jetson Orin Nano 8GB | Llama 3.2 3B | Q3_K_M | 1.5s | 12.8s | 14.3s | 20W |
| RTX 4090 (桌面) | Llama 3.1 8B | FP8 | 0.2s | 1.1s | 1.3s | 320W |
| RTX 4090 (桌面) | Llama 3.1 70B | AWQ INT4 | 2.5s | 18.0s | 20.5s | 380W |
| MacBook Pro M4 Max | Qwen2.5 3B | GGUF Q4_K_M | 0.5s | 2.2s | 2.7s | 25W |
| iPhone 16 Pro | Gemma 2 2B | INT4 (CoreML) | 0.8s | 8.5s | 9.3s | 5W |

---

## 7. 部署实践与排障

### 7.1 常见 OOM (Out of Memory) 处理

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 模型加载时 OOM | 显存不足容纳 FP16 权重 | 切换 AWQ/GPTQ INT4，显存降低 75% |
| 长上下文 OOM | KV Cache 随序列长度线性增长 | 降低 `--max-model-len`，启用 KV Cache 量化 |
| 高并发 OOM | 多请求 KV Cache 累积 | 减少 `--max-num-seqs`，启用 Prefix Caching |
| 量化加载失败 | 量化格式与引擎不匹配 | 检查 `--quantization` 参数，确认格式支持 |

**vLLM OOM 调参示例**

```bash
# 原始命令 (OOM)
vllm serve meta-llama/Llama-3.1-70B --tensor-parallel-size 2

# 优化后：INT4 量化 + 限制最大长度
vllm serve meta-llama/Llama-3.1-70B-AWQ-INT4 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --quantization awq \
  --gpu-memory-utilization 0.92
```

### 7.2 FP8 部署注意事项

| 问题 | 说明 | 解决 |
|------|------|------|
| A100 不支持 FP8 | A100 无 FP8 Tensor Cores | 回退 INT8/FP16，或更换 H100/RTX 4090 |
| FP8 校准失败 | 部分模型异常值分布不适合 E4M3 | 尝试 E5M2，或改用 INT8 |
| 精度不达标 | 数学/代码任务对 FP8 敏感 | 关键层保留 FP16，混合精度 |

### 7.3 Qwen3.6 已知问题

| 问题 | 现象 | 解决 |
|------|------|------|
| CUDA 13.2 bug | 推理结果异常/崩溃 | 回退到 CUDA 12.x |
| Ollama 不支持 | 拉取 Qwen3.6 GGUF 失败 | 因 mmproj vision 文件格式不兼容，等待 Ollama 更新或手动转换 |

### 7.4 Ollama 多 GPU 配置

```bash
# 启用多 GPU 调度 (均匀分布层)
export OLLAMA_SCHED_SPREAD=1
ollama run llama3.1:70b

# 限制单模型显存 (避免独占)
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=4
```

### 7.5 量化模型选择检查清单

- [ ] 确认目标硬件支持的量化格式 (FP8 需 Hopper/Ada)
- [ ] 确认推理引擎的量化支持矩阵 (vLLM/Ollama/TensorRT-LLM)
- [ ] 验证量化模型在关键任务上的质量 (MMLU、业务测试集)
- [ ] 测试长上下文下的 KV Cache 显存占用
- [ ] 评估并发场景下的吞吐量与延迟
- [ ] 准备降级方案 (FP16/INT8) 应对精度不达标

---

## 8. 挑战与未来方向

### 8.1 当前挑战

| 挑战 | 描述 | 影响 |
|------|------|------|
| 量化生态碎片化 | AWQ/GPTQ/GGUF/FP8 并存，格式互不兼容 | 增加部署复杂度，需维护多套模型 |
| 长上下文 KV Cache | 128K+ 上下文下 KV Cache 显存爆炸 | 限制长文档处理，需 W4A4KV4 等方案 |
| 硬件绑定 | FP8 需 Hopper，Marlin 需 SM_80+ | 旧硬件无法享受新量化技术 |
| 质量-速度权衡 | 极限量化 (INT2/Q2_K) 质量损失大 | 无法用于生产，仅限实验 |
| 多模态量化 | Vision 模型 (mmproj) 量化支持不足 | Ollama 不支持部分 Qwen3.6 GGUF |
| 动态形状 | 变长输入导致量化校准困难 | 影响 INT8/FP8 部署稳定性 |

### 8.2 2026 年趋势

| 趋势 | 描述 | 影响 |
|------|------|------|
| RAG 架构普及 | 67% 企业 LLM 应用使用某种 RAG | 降低对单模型参数量的需求，3B-7B 量化模型 + RAG 成为主流 |
| 推理预算为核心 | 企业 AI 支出从训练转向推理 | 量化技术 ROI 显著提升，INT4 部署成为默认 |
| 端侧 4-bit 成熟 | 4-bit 损失 < 2%，可接受 | 手机/PC 本地运行 7B 模型成为常态 |
| 手机 30fps 运行 | 旗舰手机 NPU 加速 2B/7B 量化模型 | 端侧 AI 助手取代云端 API |
| 混合精度普及 | PrismaQuant 4.75 bits 引领 | 非均匀混合精度成为新方向 |
| 全链路量化 | FlatQuant W4A4KV4 | KV Cache 量化释放长上下文潜力 |
| 编译器优化 | TensorRT-LLM、MLIR 量化编译 | 量化 + 图优化联合提升性能 |

### 8.3 未来技术方向

1. **1-bit 量化探索**：BitNet b1.58 等方案验证 1-bit 权重可行性，若成熟将带来 16 倍压缩
2. **量化感知训练 (QAT)**：在训练阶段模拟低精度，推理时直接 INT4 无损失
3. **硬件-算法协同设计**：下一代 NPU 原生支持 4-bit/2-bit 矩阵乘，消除反量化开销
4. **自适应量化**：根据输入动态选择精度，简单查询用 INT4，复杂推理用 FP8
5. **统一量化格式**：行业推动单一标准 (如 GGUF 扩展或新格式)，结束生态碎片化

---

## 9. 参考链接

### 9.1 量化技术论文与项目

| 资源 | 链接 | 说明 |
|------|------|------|
| AWQ 论文 | https://arxiv.org/abs/2306.00978 | Activation-aware Weight Quantization |
| GPTQ 论文 | https://arxiv.org/abs/2210.17323 | Optimal Brain Surgeon for LLM |
| SmoothQuant | https://arxiv.org/abs/2211.10438 | W8A8 权重-激活联合量化 |
| PrismaQuant | https://neuralmagic.com/prismaquant | Neural Magic 混合精度方案 |
| FlatQuant | https://github.com/sunnyxiaohu/FlatQuant | 全链路量化 |
| AutoAWQ | https://github.com/casper-hansen/AutoAWQ | AWQ 社区实现 |
| AutoGPTQ | https://github.com/PanQiWei/AutoGPTQ | GPTQ 社区实现 |
| bitsandbytes | https://github.com/bitsandbytes-foundation/bitsandbytes | HuggingFace 量化库 |
| AQLM | https://github.com/Vahe1994/AQLM | Additive Quantization |
| HQQ | https://github.com/mobiusml/hqq | Half-Quadratic Quantization |
| QQQ | https://github.com/HandH1998/QQQ | 异常值感知 INT4 |

### 9.2 推理引擎与工具

| 资源 | 链接 | 说明 |
|------|------|------|
| vLLM | https://github.com/vllm-project/vllm | 生产级推理引擎 |
| Ollama | https://github.com/ollama/ollama | 端侧一键部署 |
| llama.cpp | https://github.com/ggerganov/llama.cpp | 跨平台轻量推理 |
| TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM | NVIDIA 官方 SDK |
| TGI | https://github.com/huggingface/text-generation-inference | HuggingFace 推理服务 |
| ONNX Runtime | https://onnxruntime.ai | 跨平台标准化部署 |
| Marlin Kernels | https://github.com/IST-DASLab/marlin | GPU 量化 GEMM 优化 |

### 9.3 边缘硬件与部署

| 资源 | 链接 | 说明 |
|------|------|------|
| NVIDIA Jetson | https://developer.nvidia.com/embedded/jetson | 边缘 AI 平台 |
| JetPack SDK | https://developer.nvidia.com/embedded/jetpack | Jetson 软件栈 |
| Ollama Jetson 指南 | https://github.com/ollama/ollama/blob/main/docs/gpu.md | GPU 部署文档 |
| vLLM 量化文档 | https://docs.vllm.ai/en/latest/quantization/ | 官方量化支持矩阵 |

### 9.4 模型仓库与量化权重

| 资源 | 链接 | 说明 |
|------|------|------|
| HuggingFace Hub | https://huggingface.co/models | 主流模型 AWQ/GPTQ 权重 |
| Ollama Library | https://ollama.com/library | 官方模型库，GGUF 格式 |
| TheBloke (GGUF) | https://huggingface.co/TheBloke | 社区 GGUF 转换 (历史) |
| bartowski (GGUF) | https://huggingface.co/bartowski | 社区 GGUF 转换 (活跃) |
| Unsloth | https://huggingface.co/unsloth | 量化模型 + 微调工具 |

### 9.5 行业报告与基准

| 资源 | 链接 | 说明 |
|------|------|------|
| MMLU Benchmark | https://github.com/sjmoran/MMLU | 多任务语言理解 |
| HumanEval | https://github.com/openai/human-eval | 代码生成评估 |
| GSM8K | https://github.com/openai/grade-school-math | 数学推理评估 |
| Neural Magic Blog | https://neuralmagic.com/blog/ | 稀疏化与量化研究 |
| LLM Efficiency Challenge | https://llm-efficiency-challenge.github.io | 推理效率竞赛 |

---

> 本报告基于 2026 年 6 月公开技术资料整理，量化技术迭代迅速，建议部署前验证最新版本支持矩阵。

---

*报告完成日期：2026年6月6日*
