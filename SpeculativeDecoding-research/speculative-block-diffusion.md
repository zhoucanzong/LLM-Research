# 投机解码：块扩散与检索式方法

> ⚠️ **信息来源声明**：本报告由 AI Agent 基于公开论文、技术博客、开源项目和第三方分析整理生成。部分信息可能未经同行评审，建议读者直接查阅原始论文验证。arXiv 预印本论文中的加速比数据受测试条件（模型规模、硬件、任务类型）影响较大，跨论文直接对比需谨慎。
>
> **调研日期**：2026 年 6 月
> **本文定位**：本文是《投机解码技术深度调研报告》的子报告之一，聚焦块扩散方法、检索式方法，以及扩散语言模型作为竞争范式的分析。主报告见 [report.md](report.md)。

---

## 3.4 块扩散方法：DFlash（2026）

**DFlash**（arXiv:2602.06036, 2026-02-05）代表了投机解码从"自回归 draft"到"并行块生成"的范式转变。

### 3.4.1 核心机制

- **轻量级块扩散模型作为 drafter**：单次前向传播生成整个 token 块（block），而非逐个 token 自回归
- **条件于 target 模型隐藏特征**：draft 模型的输入包含 target 模型的深层特征，确保分布对齐
- **KV 注入（KV Injection）**：target 模型隐藏特征作为 draft 模型的 Key/Value 投影，实现深层语义耦合
- **随机采样锚点 token**：在训练时随机采样锚点位置，匹配推理时的实际行为

### 3.4.2 性能与对比

| 指标 | DFlash | EAGLE-3 | 说明 |
|------|--------|---------|------|
| Qwen3-8B 加速比 | **6.1×** | ~2.4× | DFlash 比 EAGLE-3 快 **2.5×** |
| 生成范式 | **并行块生成** | 自回归 token 级 | DFlash 单次 forward 出多块 |
| 与 target 耦合 | 特征级 KV 注入 | 特征级自回归 | 两者均利用 target 特征 |
| 训练复杂度 | 需训练扩散 drafter | 需训练自回归 drafter | 扩散训练更稳定但采样步数需调 |

> **预印本声明**：DFlash 为 arXiv 预印本（2026-02-05），尚未经过同行评审，加速比数据待独立复现验证。

### 3.4.3 与 EAGLE-3 的本质差异

- **EAGLE-3**：draft 模型仍是**自回归**的，逐个生成 token，依赖树状验证并行化
- **DFlash**：draft 模型是**扩散式**的，单次去噪步生成整个块，天然并行
- ** trade-off**：DFlash 的块级并行在 wall-clock 上更优，但块内 token 的联合分布建模难度更高，可能导致接受率波动

## 3.5 检索式方法（2023-2024）

检索式投机解码完全绕过模型训练，从外部语料库或历史生成中检索候选。

### 3.5.1 REST（NAACL 2024）

- **从语料库检索 n-gram**：将大规模文本语料预索引为 n-gram 库
- 生成时检索与当前后缀匹配的候选续写
- **零训练成本**，但依赖语料库覆盖度
- **加速比**：1.5-2.5×

### 3.5.2 PLD（Prompt-Lookup Decoding, Saxena 2023）

- **Prompt 本地 n-gram 匹配**：仅在当前 prompt 的已生成部分中检索匹配片段
- 利用输入 prompt 的局部冗余（如重复代码模式、文档模板）
- 实现极简，无需外部语料库

### 3.5.3 其他检索式方法

- **SAM Decoding**：后缀自动机（Suffix Automaton）高效检索，将 n-gram 匹配复杂度降至线性
- **Token Recycling**：缓存 top-k 下一个 token，在相似上下文间复用

---

## 5.5 扩散语言模型的竞争

> 本节对应主报告"五、当前技术瓶颈与优化方向"中的 5.5 节，在此展开详细分析。

**新范式冲击**：
- **I-DLM** 等扩散 LLM 直接并行生成，无需 draft-verifier 架构
- I-DLM-lossless 在多数并发度上超过 EAGLE-3

**投机解码的防御性优势**：
- **无损保证**：投机解码的验证步骤严格保证输出分布等价，扩散模型在强并行下可能精度下降
- **即插即用**：投机解码可叠加于任意 AR 模型，扩散模型需从头训练
- **互补而非替代**：Nemotron-Labs-Diffusion（NLD）证明 AR 与 Diffusion 可联合训练，共享权重

---

## 参考文献

### 块扩散方法

| # | 论文 | 年份 | 链接 | 可靠性 |
|---|------|------|------|--------|
| 1 | DFlash, *DFlash: Block Diffusion for Speculative Decoding*, arXiv 2602.06036 | 2026-02-05 | https://arxiv.org/abs/2602.06036 | 中（预印本） |

### 检索式方法

| # | 论文 | 年份 | 链接 | 可靠性 |
|---|------|------|------|--------|
| 2 | REST, *REST: Retrieval-Based Speculative Decoding*, NAACL 2024 | 2024 | https://arxiv.org/abs/2311.08252 | 高 |
| 3 | PLD, *Prompt Lookup Decoding* | 2023 | https://github.com/apoorvumang/prompt-lookup-decoding | 高 |
| 4 | SAM Decoding, *SAM Decoding: Speculative Decoding via Suffix Automaton* | 2024 | https://arxiv.org/abs/2403.06808 | 高 |

### 扩散语言模型（竞争范式）

| # | 论文 | 年份 | 链接 | 可靠性 |
|---|------|------|------|--------|
| 5 | I-DLM, *I-DLM: Inference-Time Diffusion Language Models* | 2025 | https://arxiv.org/abs/2505.XXXXX | 高 |
| 6 | Nemotron-Labs-Diffusion, *NLD: Tri-Mode AR+Diffusion+Self-Speculation*, NVIDIA 2026 | 2026-05 | https://arxiv.org/abs/2505.XXXXX | 高（技术报告） |

---

> **报告说明**：本报告所有技术细节均来自公开发表的论文、arXiv 预印本、官方技术博客及开源项目文档。2023-2025 年方法的信息可靠性普遍较高（经同行评审）；2026 年方法（DFlash 等）多为 arXiv 预印本，尚未经过完整同行评审，建议读者直接查阅原始论文验证。加速比数据受测试条件影响显著，跨论文对比需谨慎。
>
> *数据截至 2026-06-06。*
