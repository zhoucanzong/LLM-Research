# Qwen-VL 视觉语言模型系列发展脉络

## 概述

Qwen-VL（通义千问视觉语言模型）系列是阿里巴巴通义实验室推出的多模态大模型系列，历经多代迭代，从 2023 年的 Qwen-VL 发展至 2026 年的 Qwen3.7-Plus。该系列在视觉编码器架构、分辨率处理、视觉-语言融合方式、位置编码、视频理解、训练策略和 Agent 能力方面均有显著演进。

> 最后更新：2026年6月（增加Qwen3-VL、Qwen3.7-Plus信息）

---

## Qwen-VL (2023)

### 发布时间

2023 年 8 月（arXiv: 2308.12966）

### 架构详解

#### 整体架构

Qwen-VL 采用经典的三模块架构设计：**视觉编码器 → 视觉-语言适配器 → 大语言模型**。

```
输入图像 (448×448)
    │
    ▼
┌─────────────────────────┐
│  Visual Encoder          │
│  (OpenCLIP ViT-bigG)     │
│  Patch Size: 14×14       │
│  输出: 1024 visual tokens │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Vision-Language Adapter  │
│  (Single-layer Cross-Attn)│
│  256 learnable queries    │
│  输出: 256 tokens         │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  LLM (Qwen-7B)           │
│  文本 + 视觉 tokens 拼接  │
└─────────────────────────┘
```

#### 视觉编码器

- **架构**: 基于 OpenCLIP 的 ViT-bigG（Vision Transformer - Giant）
- **预训练**: 使用 OpenCLIP 的预训练权重初始化
- **输入分辨率**: 固定 448×448 像素
- **Patch 大小**: 14×14
- **输出**: 产生 (448/14)² = 1024 个视觉 token

#### 视觉-语言适配器（Position-aware Vision-Language Adapter）

- **类型**: 单层交叉注意力机制（类似 Perceiver Resampler）
- **核心设计**:
  - 使用 256 个可学习的 query 向量
  - 通过交叉注意力将 1024 个视觉 token 压缩为固定的 256 个 token
  - 在 query-key 对中引入 **2D 绝对位置编码**，保留空间位置信息
- **初始化**: 随机初始化（非预训练）
- **作用**: 在压缩视觉信息的同时保持位置感知能力，使模型具备视觉定位能力

#### 大语言模型

- **基座**: Qwen-7B（70 亿参数）
- **初始化**: 使用 Qwen-7B 预训练权重

### 训练策略

采用三阶段渐进式训练：

| 阶段 | 训练目标 | 可训练参数 | 数据 |
|------|---------|-----------|------|
| Stage 1: 预训练 | 视觉-语言对齐 | ViT + Adapter | 大规模图文对数据 |
| Stage 2: 多任务预训练 | 增强多任务能力 | 全部参数 | 交错图文数据、定位数据、OCR 数据 |
| Stage 3: 指令微调 | 对话与指令跟随 | 全部参数（冻结 ViT） | 高质量指令数据 |

### 核心特点

- **固定分辨率**: 所有图像统一缩放到 448×448
- **固定 token 数**: 无论图像内容如何，视觉 token 数量恒为 256
- **视觉定位**: 通过 adapter 中的 2D 位置编码实现 bounding box 输出
- **多图理解**: 支持多图交错输入
- **OCR 能力**: 通过训练数据获得文字识别能力

### 局限性

- 固定分辨率导致高分辨率图像信息丢失
- 固定 256 token 压缩可能丢失细粒度视觉信息
- Cross-attention resampler 引入额外参数但信息瓶颈明显
- 仅有 7B 单一尺寸

---

## Qwen2-VL (2024)

### 发布时间

2024 年 9 月（arXiv: 2409.12191）

### 架构详解

#### 整体架构

Qwen2-VL 进行了革命性的架构重设计，引入 **Naive Dynamic Resolution** 和 **M-RoPE** 两大核心创新。

```
输入图像 (任意分辨率)
    │
    ▼
┌──────────────────────────────┐
│  Visual Encoder (ViT ~675M)   │
│  - 2D-RoPE (替代绝对位置编码)  │
│  - 支持任意分辨率输入          │
│  输出: H/14 × W/14 tokens     │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Token Compression (MLP)      │
│  合并相邻 2×2 tokens → 1 token │
│  输出: H/28 × W/28 tokens     │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  LLM (Qwen2 系列)             │
│  + M-RoPE 位置编码             │
│  (temporal, height, width)    │
└──────────────────────────────┘
```

#### 视觉编码器

- **架构**: 重新设计的 ViT，约 **675M 参数**
- **关键创新**: 所有模型尺寸（2B、7B、72B）共享同一个 675M ViT
- **预训练初始化**: 从 **DFN（Data Filtering Networks）** 初始化
- **位置编码**: 使用 **2D-RoPE** 完全替代传统绝对位置嵌入
  - 使 ViT 能够处理任意分辨率输入
  - 自然编码 2D 空间关系
- **Patch 大小**: 14×14

#### Naive Dynamic Resolution（朴素动态分辨率）

- **核心思想**: 不对图像做任何分辨率归一化或分割，直接以原始分辨率（或按比例缩放后的分辨率）输入 ViT
- **实现方式**:
  - 移除 ViT 中的绝对位置嵌入，改用 2D-RoPE
  - 视觉 token 数量与图像尺寸成正比：`(H/14) × (W/14)`
  - MLP 层将相邻 2×2 token 合并为 1 个，最终 token 数为 `(H/28) × (W/28)`
- **优势**: 保留图像原始纵横比和分辨率信息，避免信息失真

#### M-RoPE（Multimodal Rotary Position Embedding）

这是 Qwen2-VL 最重要的位置编码创新，将旋转位置编码分解为三个独立维度：

```
M-RoPE = [Temporal_RoPE | Height_RoPE | Width_RoPE]

对于文本 token:
  temporal_id = height_id = width_id = position (三者同步递增)

对于图像 token:
  temporal_id = 常量 (图像无时间维度)
  height_id = 行位置
  width_id = 列位置

对于视频 token:
  temporal_id = 帧索引 (随帧变化)
  height_id = 帧内行位置
  width_id = 帧内列位置
```

- **设计动机**: 统一文本、图像、视频三种模态的位置表示
- **实现**: 将 RoPE 的频率维度均分为三份，分别编码时间、高度、宽度信息
- **效果**: 模型能够自然理解多模态序列中每个 token 的空间-时间位置

#### 视频处理

- **帧采样**: 固定 2 FPS（每秒 2 帧）
- **3D 卷积**: 在 ViT 后使用 depth=2 的 3D 卷积，将相邻 2 帧在时间维度合并
- **token 上限**: 每个视频最多 16384 个视觉 token
- **时序建模**: 通过 M-RoPE 的 temporal 维度实现帧间关系建模

#### Token 压缩机制

- **方式**: MLP 投影层（非 cross-attention）
- **压缩比**: 4:1（2×2 → 1）
- **优势**: 相比 Qwen-VL 的 cross-attention resampler，MLP 投影更简洁高效，且保留了动态 token 数量的特性

### 训练策略

三阶段训练，总计约 **1.4T tokens**：

| 阶段 | 训练内容 | 可训练参数 | 数据规模 |
|------|---------|-----------|---------|
| Stage 1 | 视觉编码器预训练 | 仅 ViT | ~600B tokens |
| Stage 2 | 全参数联合训练 | ViT + LLM 全部 | 总计 ~1.4T tokens |
| Stage 3 | 指令微调 | LLM（冻结 ViT） | 高质量指令数据 |

### 模型尺寸

| 模型 | LLM 参数 | ViT 参数 | 总参数 |
|------|---------|---------|--------|
| Qwen2-VL-2B | 1.5B | 675M | ~2.2B |
| Qwen2-VL-7B | 7.6B | 675M | ~8.3B |
| Qwen2-VL-72B | 72B | 675M | ~72.7B |

### 核心变更（相比 Qwen-VL）

1. **分辨率**: 固定 448×448 → 任意动态分辨率
2. **位置编码**: 2D 绝对位置 → 2D-RoPE (ViT) + M-RoPE (LLM)
3. **视觉-语言融合**: Cross-attention resampler → MLP 线性投影
4. **token 数量**: 固定 256 → 动态（与图像尺寸成正比）
5. **ViT 初始化**: OpenCLIP ViT-bigG → DFN 初始化的 675M ViT
6. **模型系列**: 单一 7B → 2B/7B/72B 三个尺寸
7. **视频理解**: 新增原生视频理解能力（3D 卷积 + M-RoPE temporal）

### 创新亮点

- **Naive Dynamic Resolution** 是业界首创的"零归一化"动态分辨率方案
- **M-RoPE** 优雅地统一了文本/图像/视频的位置表示
- 675M ViT 跨所有模型尺寸共享，提高训练效率
- Agent 能力涵盖 UI 操作、机器人控制、导航、游戏等

---

## Qwen2.5-VL (2025)

### 发布时间

2025 年 1 月（arXiv: 2502.13923）

### 架构详解

#### 整体架构

Qwen2.5-VL 在 Qwen2-VL 基础上进一步优化视觉编码器效率，引入 **Window Attention** 和 **绝对时间编码**。

```
输入图像/视频 (原生分辨率, 动态 FPS)
    │
    ▼
┌───────────────────────────────────┐
│  Visual Encoder (ViT, 从头训练)     │
│  - 2D-RoPE                         │
│  - Window Attention (大部分层)       │
│    + Full Attention (仅 4 层)       │
│  - RMSNorm + SwiGLU                │
│  - 最大窗口: 8×8                    │
│  输出: 原生分辨率 visual tokens      │
└───────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────┐
│  MLP-based Merger                   │
│  融合局部 patch 特征                 │
│  压缩 token 数量                    │
└───────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────┐
│  LLM (Qwen2.5 系列)                │
│  + 增强 M-RoPE                     │
│    (绝对时间编码, 动态 FPS)          │
└───────────────────────────────────┘
```

#### 视觉编码器（核心重设计）

- **训练方式**: 完全从头训练（Trained from Scratch），不再依赖外部预训练权重
- **位置编码**: 2D-RoPE（延续 Qwen2-VL）
- **注意力机制**: **Window Attention + Full Attention 混合设计**
  - 大部分层使用 Window Attention（窗口大小最大 8×8）
  - 仅 **4 层使用 Full Attention**
  - 小于 8×8 的区域不需要 padding，保持原生分辨率
  - 将复杂度从 O(n²) 降低至接近线性 O(n)
- **归一化**: 使用 **RMSNorm**（替代 LayerNorm），与 LLM 架构一致
- **激活函数**: 使用 **SwiGLU**（替代传统 FFN），与 LLM 架构一致
- **设计理念**: ViT 架构向 LLM 靠拢，统一模块设计

#### 原生动态分辨率（增强版）

- 延续 Qwen2-VL 的 Naive Dynamic Resolution 思想
- 图像按原生分辨率处理，无归一化
- 输出 token 数量完全由图像实际尺寸决定

#### MLP-based Merger

- **功能**: 融合局部 patch 特征并压缩 token 数量
- **设计**: 基于 MLP 的特征融合（非 cross-attention）
- 延续 Qwen2-VL 的 2×2 → 1 压缩策略

#### 位置编码创新

**绝对时间编码（Absolute Time Encoding）**:
- 将 M-RoPE 的 temporal 维度 ID 与**真实时间（秒）**对齐
- 不再使用帧索引作为时间标识
- 引入 **动态 FPS 训练**：训练时使用不同帧率采样
- mRoPE ID 直接与时间流速对应

**绝对坐标表示（Absolute Coordinates）**:
- Bounding box 和 point 坐标使用**图像原始像素坐标**（非归一化到 [0,1]）
- 模型直接学习图像的真实尺度
- 输出格式: `{"bbox_2d": [x1, y1, x2, y2]}` 使用绝对像素值

#### 视频处理

- **动态 FPS**: 根据视频内容动态决定采样帧率（非固定 2 FPS）
- **超长视频**: 支持小时级别长视频理解
- **时间定位**: 支持秒级事件定位（Temporal Video Grounding）
- **绝对时间**: 使用 `mm:ss.ff` 格式精确定位视频片段

### 训练策略

多阶段训练流程：

| 阶段 | 训练目标 | 说明 |
|------|---------|------|
| Stage 1: CLIP 预训练 | 视觉编码器基础能力 | 从头训练 ViT 的视觉表示能力 |
| Stage 2: 视觉-语言对齐 | 多模态对齐 | 将视觉特征与语言空间对齐 |
| Stage 3: 端到端训练 | 联合优化 | 全参数端到端训练 |
| Stage 4: 后训练 | 偏好对齐 | SFT + DPO（直接偏好优化） |

### 模型尺寸

| 模型 | 参数规模 | 说明 |
|------|---------|------|
| Qwen2.5-VL-3B | ~3B | 边缘端部署方案 |
| Qwen2.5-VL-7B | ~7B | 超越 GPT-4o-mini |
| Qwen2.5-VL-72B | ~72B | 旗舰模型，对标 GPT-4o |

### 核心变更（相比 Qwen2-VL）

1. **ViT 训练**: DFN 初始化 → 从头训练
2. **注意力机制**: Full Attention → Window Attention + Full Attention 混合（仅 4 层 Full Attn）
3. **ViT 归一化/激活**: 传统设计 → RMSNorm + SwiGLU（与 LLM 统一）
4. **时间编码**: 帧索引 → 绝对时间（真实秒数）
5. **帧率策略**: 固定 2 FPS → 动态 FPS 训练
6. **坐标表示**: 归一化坐标 → 绝对像素坐标
7. **后训练**: 新增 DPO 阶段
8. **模型尺寸**: 2B/7B/72B → 3B/7B/72B

### 创新亮点

- **Window Attention** 大幅降低 ViT 计算复杂度，实现接近线性的扩展性
- **绝对时间编码** 使模型真正理解视频的时间流逝
- **绝对坐标** 让模型直接感知图像物理尺度
- **从头训练 ViT** 摆脱对外部预训练的依赖，实现端到端最优
- 强大的 **Agent 能力**：支持 Computer Use 和 Phone Use
- **文档解析**: 独创 QwenVL HTML 格式，支持杂志、论文、网页等复杂文档

---

## 跨代架构对比总结

### 视觉编码器演进

| 维度 | Qwen-VL (2023) | Qwen2-VL (2024) | Qwen2.5-VL (2025) |
|------|----------------|-----------------|-------------------|
| 架构 | OpenCLIP ViT-bigG | 自定义 ViT ~675M | 重新设计 ViT (从头训练) |
| 初始化 | OpenCLIP 预训练 | DFN 预训练 | 从头训练 (CLIP→对齐→E2E) |
| 位置编码 | 绝对位置嵌入 | 2D-RoPE | 2D-RoPE |
| 注意力 | Full Attention | Full Attention | Window Attn + Full Attn (仅4层) |
| 归一化 | LayerNorm | LayerNorm | RMSNorm |
| 激活函数 | GELU | GELU | SwiGLU |
| 计算复杂度 | O(n²) | O(n²) | ~O(n) (近线性) |

### 分辨率处理演进

| 维度 | Qwen-VL | Qwen2-VL | Qwen2.5-VL |
|------|---------|----------|------------|
| 策略 | 固定 448×448 | Naive Dynamic Resolution | 原生动态分辨率 (增强) |
| Token 数 | 固定 256 | 动态 (H/28 × W/28) | 动态 (原生分辨率) |
| 纵横比 | 不保留 | 保留 | 保留 |
| 坐标表示 | — | 归一化坐标 | 绝对像素坐标 |

### 视觉-语言融合演进

| 维度 | Qwen-VL | Qwen2-VL | Qwen2.5-VL |
|------|---------|----------|------------|
| 方式 | Cross-Attention Resampler | MLP (2×2→1) | MLP-based Merger |
| 压缩比 | 1024→256 (4:1) | 4:1 (空间) | 4:1 (空间) |
| 复杂度 | 高 (交叉注意力) | 低 (线性投影) | 低 (线性投影) |
| 信息保留 | 固定瓶颈 | 动态保留 | 动态保留 + 局部融合 |

### 位置编码演进

| 维度 | Qwen-VL | Qwen2-VL | Qwen2.5-VL |
|------|---------|----------|------------|
| ViT 内部 | 绝对位置嵌入 | 2D-RoPE | 2D-RoPE |
| LLM 内部 | 标准 RoPE | M-RoPE (temporal+H+W) | 增强 M-RoPE + 绝对时间编码 |
| Adapter | 2D 绝对位置编码 | — (MLP无位置) | — (MLP无位置) |
| 时间建模 | 无 | 帧索引 (M-RoPE temporal) | 绝对秒数 (动态FPS) |

### 视频理解演进

| 维度 | Qwen-VL | Qwen2-VL | Qwen2.5-VL |
|------|---------|----------|------------|
| 支持 | 不支持 | 原生支持 | 原生支持 (增强) |
| 帧率 | — | 固定 2 FPS | 动态 FPS |
| 时序合并 | — | 3D 卷积 (depth=2) | 3D 卷积 + 动态FPS |
| 时间表示 | — | 帧索引 | 绝对时间 (秒) |
| 最大时长 | — | 受 16384 token 限制 | 小时级长视频 |
| 时间定位 | — | 不支持 | 秒级事件定位 |

### 训练策略演进

| 维度 | Qwen-VL | Qwen2-VL | Qwen2.5-VL |
|------|---------|----------|------------|
| 训练阶段 | 3 阶段 | 3 阶段 | 4 阶段 (含 DPO) |
| ViT 训练 | 冻结预训练 ViT | 从 Stage 1 训练 ViT (600B tokens) | 从头训练 (CLIP预训练) |
| 总数据量 | — | ~1.4T tokens | — |
| 后训练 | SFT | SFT | SFT + DPO |
| 对齐方式 | — | — | 直接偏好优化 |

### Agent 能力演进

| 维度 | Qwen-VL | Qwen2-VL | Qwen2.5-VL |
|------|---------|----------|------------|
| 视觉定位 | Bounding box | Bounding box | Bbox + Point + 绝对坐标 |
| OCR | 基础文字识别 | 增强 OCR | 多场景多语言多方向 OCR |
| UI 操作 | 不支持 | 基础支持 | Computer Use + Phone Use |
| 文档解析 | 不支持 | 基础支持 | QwenVL HTML 格式 |
| 结构化输出 | 不支持 | 不支持 | JSON 结构化输出 |

---

## 技术演进总结

### 三大设计哲学转变

1. **从固定到动态**:
   - 分辨率: 固定 → 动态
   - Token 数: 固定 → 动态
   - 帧率: 无 → 固定 2FPS → 动态 FPS
   - 坐标: 无 → 归一化 → 绝对坐标

2. **从复杂到简洁**:
   - 融合方式: Cross-Attention → MLP
   - 位置编码: 绝对位置 → RoPE (相对)
   - ViT 架构: 与 LLM 统一 (RMSNorm + SwiGLU)

3. **从借用到自研**:
   - ViT 初始化: OpenCLIP → DFN → 从头训练
   - 位置编码: 借用标准方案 → M-RoPE 原创
   - 注意力: 标准 Full Attn → Window + Full 混合

### 性能里程碑

- **Qwen-VL (2023)**: 开源 VLM 中首个同时具备视觉定位、OCR、多图理解的模型
- **Qwen2-VL (2024)**: 在多项基准测试中超越 GPT-4V，M-RoPE 成为多模态位置编码的标杆方案
- **Qwen2.5-VL (2025)**: 72B 模型在多项基准上对标 GPT-4o 和 Claude 3.5 Sonnet；3B 模型超越上一代 7B

---

## Qwen3-VL（2025年5月—）

### 发布时间

2025年5月发布，技术报告于2025年11月提交（arXiv: 2511.21631）

### 核心突破

Qwen3-VL被定位为Qwen系列中“最强大的视觉语言模型”，实现三大核心支柱：

1. **纯文本理解显著增强**：在多个场景中超越同级别纯文本模型
2. **多模态能力全面升级**：图像、视频、文档理解均达到新SOTA
3. **Agent和GUI交互能力**：面向GUI/Agent工作流的原生设计

### 架构升级

- **增强的interleaved-MRoPE**：更强的图像和视频时空建模
- **原生256K交错上下文**：支持超长图文交错序列
- **Thinking模式**：Qwen3-VL-8B-Thinking等变体支持视觉推理增强

### 模型系列

| 模型 | 参数量 | 特点 |
|------|--------|------|
| Qwen3-VL-2B | ~2B | 边缘部署 |
| Qwen3-VL-8B | ~8.8B | 标准版本 |
| Qwen3-VL-8B-Thinking | ~8.8B | 推理增强版（2025年10月） |
| Qwen3-VL-72B | ~72B | 旗舰版本 |
| Qwen3-VL-Flash | 轻量 | 低成本视觉理解（2026年2月上线百炼） |

---

## Qwen3.7-Plus（2026年6月）—— 多模态Agent基座

### 概述

Qwen3.7-Plus是Qwen系列最新的多模态Agent模型，于2026年6月2日发布，在Vision Arena榜单中进入全球前五、中国第一。

### 核心能力

与Qwen3-VL的纯视觉理解不同，Qwen3.7-Plus将视觉与语言统一为一体化Agent基座：

| 能力 | 说明 |
|------|------|
| Multimodal Agent | 统一处理图像、视频、屏幕、网页和文本，在GUI/CLI/工具环境中完成任务 |
| Visual Agent | 视觉理解 + 代码解释器 + 搜索增强，解决视觉谜题和复杂推理 |
| Visual Coding | 从图像/视频生成SVG、网页、交互式前端 |
| GUI Agent | 移动端/桌面端界面理解、控件定位、任务规划和多步操作 |
| Real-world Perception | 真实场景、文档图表、OCR、视频、驾驶场景理解 |

### 技术定位

- 在Qwen3.7强大文本能力基础上全面升级视觉-语言能力
- 保持完整的编码、工具调用、调试等Agent功能
- 实现"看、想、写、做、验"的端到端智能体工作流
- 仅通过阿里云百炼API提供服务（非开源）

---

## 论文引用

1. **Qwen-VL**:
   > Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., & Zhou, J. (2023). *Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond*. arXiv:2308.12966.

2. **Qwen2-VL**:
   > Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., & Lin, J. (2024). *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution*. arXiv:2409.12191.

3. **Qwen2.5-VL**:
   > Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. (2025). *Qwen2.5-VL Technical Report*. arXiv:2502.13923.

4. **Qwen3-VL**:
   > Bai, S., Cai, J., et al. (2025). *Qwen3-VL Technical Report*. arXiv:2511.21631.

---

## 参考资源

- Qwen 官方网站: https://qwen.ai
- Qwen GitHub: https://github.com/QwenLM
- Qwen2-VL 官方博客: https://qwenlm.github.io/blog/qwen2-vl/
- Qwen2.5-VL 官方博客: https://qwenlm.github.io/blog/qwen2.5-vl/
- Qwen3-VL GitHub: https://github.com/QwenLM/Qwen3-VL
- Qwen3.7-Plus 博客: https://qwen.ai/blog?id=qwen3.7-plus
- Hugging Face 模型库: https://huggingface.co/Qwen
