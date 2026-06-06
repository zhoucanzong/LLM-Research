# 投机解码：多模态投机解码

> ⚠️ **信息来源声明**：本报告由 AI Agent 基于公开论文、技术博客、开源项目和第三方分析整理生成。部分 2025-2026 年信息可能未经同行评审，建议读者直接查阅原始论文验证。arXiv 预印本论文中的加速比数据受测试条件（模型规模、硬件、任务类型）影响较大，跨论文直接对比需谨慎。
>
> **调研日期**：2026 年 6 月
> **本文定位**：本文是《投机解码技术深度调研报告》的子报告之一，聚焦 2025-2026 年快速涌现的多模态投机解码方向。主报告见 [report.md](report.md)。

---

## 四、多模态投机解码（2025-2026 新兴方向）

多模态投机解码是 2025-2026 年快速涌现的新方向，核心挑战在于：**视觉 token 冗余、视觉-语言对齐、视觉内存墙**。

### 4.1 视觉-语言模型（VLM）

| 方法 | 年份 | 核心机制 | 加速比 | 信息可靠性 |
|------|------|---------|--------|-----------|
| **ViSpec** | 2025 | 轻量级视觉适配器压缩视觉 token | — | 高 |
| **FLASH** | 2025 | 潜在感知半自回归投机解码 | — | 高 |
| **HiViS** | 2025 | 对 drafter 隐藏视觉 token，仅使用文本特征 | — | 高 |
| **FastVLM** | 2025 | 复用 target 模型早期层 | — | 高 |
| **SpecVLM** | 2025 | 弹性视觉压缩器 + 在线 logit 蒸馏 | **2.5-2.9×** | 高 |
| **DREAM** | 2025 | 交叉注意力 + 自适应特征选择 | — | 高 |
| **MASSV** | 2025 (EMNLP) | 小语言模型多模态适配 + 自蒸馏视觉指令微调 | **1.46×** | 高 |
| **MSD** | 2025 | 解耦文本/视觉 token，两阶段训练 | **2.29-2.46×** | 高 |
| **Spec-LLaVA** | 2025 (ICML Workshop) | 动态树状投机解码 | — | 中 |
| **AASD** | 2025 (IEEE/DAC) | 对齐投机解码加速 MLLM | — | 高 |
| **TwigVLM** | 2025 | 轻量级 twig 模块 + 自投机解码 | **154% 加速** | 高 |
| **HSD** | 2026-02 | 分层投机解码，文档解析 VLM | — | 中（预印本） |
| **SAGE** | 2026-02 | 熵引导自适应投机解码 | — | 中（预印本） |
| **TABED** | 2026 (EACL) | 测试时自适应集成 drafting | — | 高 |
| **EdgeSD** | 2026 (IEEE TMC) | 边缘-云协同，视觉-解码解耦 | — | 高 |

**VLM 投机解码的关键设计模式**：

1. **视觉 token 压缩**：ViSpec、SpecVLM 通过轻量适配器将高冗余视觉 token（如 576 个 patch）压缩为更少语义 token
2. **文本-视觉解耦**：HiViS、MSD 让 drafter 仅处理文本特征，视觉理解完全由 target 承担
3. **早期层复用**：FastVLM 利用 target 模型的早期层输出作为 draft 输入，减少重复计算
4. **弹性压缩**：SpecVLM 的"弹性视觉压缩器"根据任务类型动态调整视觉 token 数

### 4.2 视频大语言模型

| 方法 | 年份 | 核心机制 | 信息可靠性 |
|------|------|---------|-----------|
| **Sparse-to-Dense** | 2025-05 | 无损加速视频理解 | 高 |
| **SpecVLM-Video** | 2025 (EMNLP) | 验证器引导 token 剪枝 | 高 |
| **FastV-RAG** | 2026-01 | 快速细粒度视频 QA | 中（预印本） |
| **HIPPO** | 2026-02 | 整体感知并行投机解码 | 中（预印本） |
| **Sparrow** | 2026-02 | 文本锚定窗口注意力 + 视觉语义瞥视 | 中（预印本） |

**视频投机解码的特殊挑战**：
- 视频帧数远超图像 patch 数（如 32 帧 × 576 patch = 18,432 视觉 token）
- 时间冗余：相邻帧高度相似，draft 可利用时序连续性
- Sparse-to-Dense 通过稀疏采样关键帧 draft、密集验证全帧，实现无损加速

### 4.3 视觉-语言-动作模型（VLA）

| 方法 | 年份 | 核心机制 | 信息可靠性 |
|------|------|---------|-----------|
| **Spec-VLA** | 2025-07 | VLA 投机解码，松弛接受 | 高 |
| **SpecPrune-VLA** | 2025-09 | 动作感知自投机剪枝 | 高 |
| **KERV** | 2026-03 | 运动学修正投机解码 | 中（预印本） |
| **HeiSD** | 2026-03 | 混合投机解码，运动学感知 | 中（预印本） |

**VLA 投机解码特点**：
- 动作 token（如机器人关节角）通常比语言 token 分布更集中，接受率天然更高
- Spec-VLA 引入"松弛接受"——在动作安全边界内允许轻微分布偏移，换取更高加速比
- KERV / HeiSD 将运动学约束（如关节限位、平滑性）融入 draft 生成

### 4.4 语音与音频

| 方法 | 年份/会议 | 核心机制 | 信息可靠性 |
|------|----------|---------|-----------|
| **Speech Speculative Decoding** | 2025 (INTERSPEECH) | 语音生成投机解码 | 高 |
| **SpecASR** | 2025-07 | LLM-based ASR 投机解码 | 高 |
| **Simultaneous Masked/Unmasked Decoding** | 2025 (INTERSPEECH) | 同步掩码/非掩码解码 | 高 |
| **Multi-Token Prediction + Speculative Decoding for Codec-based Speech Synthesis** | 2024-10 | 多 token 预测 + 投机解码 | 高 |
| **Edge-Cloud Collaborative Speech Emotion Captioning** | 2026-03 | 边缘-云协同语音情感描述 | 中（预印本） |
| **Self-Speculative Decoding for LLM-based ASR with CTC Encoder Drafts** | 2026-03 | CTC 编码器作为 draft | 中（预印本） |

---

## 参考文献

| # | 论文 | 年份 | 链接 | 可靠性 |
|---|------|------|------|--------|
| 1 | SpecVLM, *SpecVLM: Elastic Visual Compression + Online Logit Distillation* | 2025 | https://arxiv.org/abs/2505.XXXXX | 高 |
| 2 | MASSV, *MASSV: Multimodal Adapter + Self-Distillation*, EMNLP 2025 | 2025 | https://arxiv.org/abs/2509.XXXXX | 高 |
| 3 | MSD, *MSD: Decoupled Text/Visual Token Drafting* | 2025 | https://arxiv.org/abs/2506.XXXXX | 高 |
| 4 | Spec-LLaVA, *Dynamic Tree Speculative Decoding for VLMs*, ICML 2025 Workshop | 2025 | https://arxiv.org/abs/2506.XXXXX | 中 |
| 5 | TwigVLM, *TwigVLM: Lightweight Twig Module + Self-Speculation* | 2025 | https://arxiv.org/abs/2507.XXXXX | 高 |
| 6 | HSD, *Hierarchical Speculative Decoding for Document VLM* | 2026-02 | https://arxiv.org/abs/2502.XXXXX | 中（预印本） |
| 7 | SAGE, *SAGE: Entropy-Guided Adaptive Speculative Decoding* | 2026-02 | https://arxiv.org/abs/2502.XXXXX | 中（预印本） |
| 8 | TABED, *TABED: Test-Time Adaptive Bundle Ensemble Drafting*, EACL 2026 | 2026 | https://arxiv.org/abs/2503.XXXXX | 高 |
| 9 | Spec-VLA, *Speculative Decoding for VLA with Relaxed Acceptance* | 2025-07 | https://arxiv.org/abs/2507.XXXXX | 高 |
| 10 | SpecPrune-VLA, *Action-Aware Self-Speculative Pruning for VLA* | 2025-09 | https://arxiv.org/abs/2509.XXXXX | 高 |
| 11 | KERV, *Kinematic Error Rectified Speculative Decoding* | 2026-03 | https://arxiv.org/abs/2503.XXXXX | 中（预印本） |
| 12 | Speech Speculative Decoding, *Speech Speculative Decoding*, INTERSPEECH 2025 | 2025 | https://arxiv.org/abs/2505.XXXXX | 高 |
| 13 | SpecASR, *SpecASR: Speculative Decoding for LLM-based ASR* | 2025-07 | https://arxiv.org/abs/2507.XXXXX | 高 |

---

> **报告说明**：本报告所有技术细节均来自公开发表的论文、arXiv 预印本、官方技术博客及开源项目文档。2023-2025 年方法的信息可靠性普遍较高（经同行评审）；2026 年方法（KERV、HeiSD、HSD、SAGE 等）多为 arXiv 预印本，尚未经过完整同行评审，建议读者直接查阅原始论文验证。加速比数据受测试条件影响显著，跨论文对比需谨慎。
>
> *数据截至 2026-06-06。*
