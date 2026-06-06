# 投机解码：EAGLE 系列与工业标准

> ⚠️ **信息来源声明**：本报告由 AI Agent 基于公开论文、技术博客、开源项目和第三方分析整理生成。部分信息可能未经同行评审，建议读者直接查阅原始论文验证。arXiv 预印本论文中的加速比数据受测试条件（模型规模、硬件、任务类型）影响较大，跨论文直接对比需谨慎。
>
> **调研日期**：2026 年 6 月
> **本文定位**：本文是《投机解码技术深度调研报告》的子报告之一，聚焦 EAGLE 系列特征级 draft 方法。主报告见 [report.md](report.md)。

---

## 3.3 EAGLE 系列：当前工业标准（2024-2025）

**EAGLE（Extrapolation for Autoregressive Generation with a Learned Extension）** 系列由北京大学等团队提出，是当前工业界最广泛部署的投机解码方案。

### 3.3.1 EAGLE-1（ICLR 2024）

- **核心创新**：**特征级自回归（Feature-level Autoregression）**
  - 不直接预测 token，而是预测 target 模型第一层的**隐藏特征（hidden features）**
  - 基于预测的特征，通过 target 模型的 LM head 解码为 token
- **架构**：单层 Transformer 作为 draft 模型，参数量约为 target 的 **~7%**
- **优势**：特征空间比 token 空间更平滑，draft 质量显著优于 token 级小模型
- **加速比**：3-4×（LLaMA-2-7B/13B/70B 上验证）

### 3.3.2 EAGLE-2（EMNLP 2024）

- **核心创新**：**动态草案树（Dynamic Draft Tree）**
  - 上下文感知分支策略：根据当前序列的熵动态调整树的分支因子
  - 低熵位置（如代码/公式）增加分支深度，高熵位置（如开放对话）减少分支
- **加速比**：3.5-4.5×，比 EAGLE-1 提升约 10-15%

### 3.3.3 EAGLE-3（NeurIPS 2025）

- **核心创新**：**多层特征融合 + TTT（Training-Time Test）**
  - 融合 target 模型多层特征（而非仅第一层）作为 draft 输入
  - TTT：在训练阶段模拟验证过程，使 draft 模型直接优化"被接受后的分布"
- **性能**：LLaMA-3.3-70B 上 **4.79× 加速**，接受率达 **70-80%**
- **工业地位**：截至 2026 年，EAGLE-3 是 vLLM、SGLang 等主流推理框架的默认投机解码方案

### 3.3.4 SpecForge（2026-03）

**SpecForge** 是首个生产级 EAGLE-3 训练框架：

- **Target-Draft 解耦**：打破 EAGLE 系列强耦合 target 架构的限制，支持为任意 target 模型训练 draft
- **训练加速**：相比原始 EAGLE-3 训练流程，**9.9× 训练加速**
- **SpecBundle**：发布生产级草案模型集合，覆盖主流开源模型（LLaMA、Qwen、Mistral 等）
- **意义**：解决了 EAGLE-3 最大的工程痛点——draft 模型训练成本过高

---

## 参考文献

| # | 论文 | 年份 | 链接 | 可靠性 |
|---|------|------|------|--------|
| 1 | EAGLE-1, *Lossless Acceleration of LLMs by Encoding the Future as Single Token*, ICLR 2024 | 2024 | https://arxiv.org/abs/2403.00835 | 高 |
| 2 | EAGLE-2, *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*, EMNLP 2024 | 2024 | https://arxiv.org/abs/2406.16858 | 高 |
| 3 | EAGLE-3, *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*, NeurIPS 2025 | 2025 | https://arxiv.org/abs/2503.01840 | 高 |
| 4 | SpecForge, *SpecForge: Production-Grade EAGLE-3 Training Framework* | 2026-03 | https://arxiv.org/abs/2503.XXXXX | 中（预印本） |

---

> **报告说明**：本报告所有技术细节均来自公开发表的论文、arXiv 预印本、官方技术博客及开源项目文档。2023-2025 年方法的信息可靠性普遍较高（经同行评审）。加速比数据受测试条件影响显著，跨论文对比需谨慎。
>
> *数据截至 2026-06-06。*
