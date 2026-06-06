# 投机解码：经典方法与自投机

> ⚠️ **信息来源声明**：本报告由 AI Agent 基于公开论文、技术博客、开源项目和第三方分析整理生成。部分信息可能未经同行评审，建议读者直接查阅原始论文验证。arXiv 预印本论文中的加速比数据受测试条件（模型规模、硬件、任务类型）影响较大，跨论文直接对比需谨慎。
>
> **调研日期**：2026 年 6 月
> **本文定位**：本文是《投机解码技术深度调研报告》的子报告之一，聚焦经典独立模型 draft 与自投机方法。主报告见 [report.md](report.md)。

---

## 3.1 经典方法（2023）

### 3.1.1 Leviathan et al. (ICML 2023) / Chen et al. (2023)

2023 年，Google Research 的 **Leviathan et al.**（ICML 2023）和 DeepMind 的 **Chen et al.**（同期独立工作）几乎同时提出了投机解码的核心范式：

- **Leviathan et al.**：*Fast Inference from Transformers via Speculative Decoding*（ICML 2023）
  - 使用独立训练的小 Transformer 作为 draft model
  - 在 T5-XXL (11B) 上实现 **2-3× 加速**，draft 模型为 T5-small (60M)
  - 关键发现：draft 模型只需与 target 模型"同系列"（相同 tokenizer、相似训练数据），无需严格对齐

- **Chen et al.**：*Accelerating Large Language Model Decoding with Speculative Sampling*（2023）
  - 同样采用 draft-then-verify 框架
  - 强调接受-拒绝采样的数学正确性证明
  - 在 Chinchilla 系列模型上验证加速效果

**共同瓶颈**：
1. **分布失配（Distribution Mismatch）**：独立训练的 draft 模型难以精确匹配 target 模型的条件分布，尤其在高熵/开放式生成场景接受率骤降
2. **模型依赖**：需要为每个 target 模型准备对应的小变体，许多开源/商业模型无现成小版本
3. **内存开销**：draft 模型独立的 embedding / LM head / KV Cache 增加显存压力

## 3.2 自投机方法（2024）

2024 年，社区转向"**自投机（Self-Speculative）**"路线——不依赖独立小模型，而是让目标模型自身或其轻量扩展生成 draft。

### 3.2.1 Medusa：多头并行预测（ICML 2024）

**Medusa** 在目标模型之上添加多个并行的"预测头"（Prediction Heads），每个头负责预测未来 $k$ 步的 token：

- **架构**：冻结 target backbone，添加 $K$ 个轻量级 LM heads（通常 1-2 层 MLP）
- **树状注意力（Tree Attention）**：将多头预测的候选组合成树状结构，通过单次树状注意力验证整棵树
- **训练**：仅需微调预测头，backbone 冻结
- **加速比**：2.5-3.5×（Vicuna-7B/13B 上验证）

**局限**：多头独立预测，未建模头之间的顺序依赖，长 horizon 上 draft 质量衰减较快。

### 3.2.2 Hydra：顺序依赖草案头（NeurIPS 2024）

**Hydra** 改进 Medusa 的独立头设计，引入**顺序依赖**：

- 第 $k$ 个草案头的输入包含前 $k-1$ 个头的预测结果
- 通过自回归耦合增强 draft 的连贯性
- **加速比**：2.8-3.8×，在 Medusa 基础上有 10-20% 提升

### 3.2.3 Lookahead Decoding：完全免训练（NeurIPS 2024）

**Lookahead** 是首个**完全免训练**的自投机方法：

- **Jacobi 迭代**：将自回归生成视为定点迭代问题，通过 Jacobi 方法并行更新多个位置
- **n-gram 池**：维护一个动态 n-gram 缓存池，从历史生成中检索匹配片段作为 draft
- **零训练成本**：无需任何微调或额外模型
- **加速比**：1.5-2.5×，受限于 n-gram 匹配命中率

### 3.2.4 其他早期退出方法

- **LayerSkip**：利用早期层输出进行 draft 生成，跳过部分中间层
- **Kangaroo**：类似思路，在特定层退出并生成 draft

---

## 参考文献

### 基础方法

| # | 论文 | 年份 | 链接 | 可靠性 |
|---|------|------|------|--------|
| 1 | Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 | 2023 | https://arxiv.org/abs/2211.17192 | 高 |
| 2 | Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* | 2023 | https://arxiv.org/abs/2302.01318 | 高 |

### 自投机方法

| # | 论文 | 年份 | 链接 | 可靠性 |
|---|------|------|------|--------|
| 3 | Medusa, *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, ICML 2024 | 2024 | https://arxiv.org/abs/2401.10774 | 高 |
| 4 | Hydra, *Hydra: Sequence-Wise Self-Speculative Decoding*, NeurIPS 2024 | 2024 | https://arxiv.org/abs/2402.02524 | 高 |
| 5 | Lookahead Decoding, *Lookahead Decoding: Multi-Token Inference via Jacobi Iteration*, NeurIPS 2024 | 2024 | https://arxiv.org/abs/2402.02057 | 高 |
| 6 | LayerSkip, *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding* | 2024 | https://arxiv.org/abs/2404.16710 | 高 |

---

> **报告说明**：本报告所有技术细节均来自公开发表的论文、arXiv 预印本、官方技术博客及开源项目文档。2023-2025 年方法的信息可靠性普遍较高（经同行评审）。加速比数据受测试条件影响显著，跨论文对比需谨慎。
>
> *数据截至 2026-06-06。*
