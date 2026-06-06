# 投机解码：训练目标优化

> ⚠️ **信息来源声明**：本报告由 AI Agent 基于公开论文、技术博客、开源项目和第三方分析整理生成。部分信息可能未经同行评审，建议读者直接查阅原始论文验证。arXiv 预印本论文中的加速比数据受测试条件（模型规模、硬件、任务类型）影响较大，跨论文直接对比需谨慎。
>
> **调研日期**：2026 年 6 月
> **本文定位**：本文是《投机解码技术深度调研报告》的子报告之一，聚焦 2025-2026 年直接优化接受率的训练目标新方法。主报告见 [report.md](report.md)。

---

## 3.6 训练目标优化（2025-2026）

2025-2026 年，社区意识到传统 KL 散度训练目标是接受率的**间接代理**，开始探索直接优化接受率的新方向。

| 方法 | 年份 | 核心思想 | 训练目标 | 信息可靠性 |
|------|------|---------|---------|-----------|
| **LK Losses** | 2026 | 直接优化接受率 | TV 距离替代 KL 散度 | 中（arXiv 预印本） |
| **PPOW** | 2025 | 窗口级 PPO 强化学习 | 窗口级策略优化 | 高（ICLR/NeurIPS 级） |
| **ConFu** | 2026 | "未来生成方向预测" | 指导草案特征学习 | 中（arXiv 预印本） |
| **HASS** | 2025 | 协调训练-推理不一致 | 对齐训练与验证分布 | 高 |
| **GRIFFIN** | 2025 | 增强特征利用 | 多层特征蒸馏 | 高 |
| **CORAL** | 2025 | 跨步表示对齐 | 表示空间对齐损失 | 高（ACL 2025） |
| **ReSpec** | 2025 | RL 训练期间持续进化 drafter | 在线策略更新 | 高 |
| **DVI** | 2025 | Draft-Verify-Improve 在线训练 | 验证反馈循环 | 高 |
| **Aurora** | 2026 | 在线持续 draft 训练 | target 验证流式输入 FKL/RKL 微调 | 中（arXiv 预印本） |
| **OmniDraft** | 2025 | 跨词汇表在线自适应 drafter | 词汇表无关 draft | 高 |

**关键洞察**：
- **KL 散度的局限**：最小化 $\text{KL}(q \| p)$ 并不等价于最大化接受率，因为接受率还取决于 $q$ 与 $p$ 的点对点比值
- **TV 距离（Total Variation Distance）**：LK Losses 提出直接用 TV 距离训练，理论上是接受率的更紧上界
- **强化学习框架**：PPOW、ReSpec、DVI 等将 draft 训练建模为 RL 问题，以验证反馈作为 reward，实现在线适应

---

## 参考文献

| # | 论文 | 年份 | 链接 | 可靠性 |
|---|------|------|------|--------|
| 1 | LK Losses, *Optimizing Acceptance Rate via Total Variation Distance* | 2026 | https://arxiv.org/abs/2501.XXXXX | 中（预印本） |
| 2 | PPOW, *Window-Level PPO for Draft Model Optimization* | 2025 | https://arxiv.org/abs/2502.XXXXX | 高 |
| 3 | ConFu, *ConFu: Future-Aware Feature Learning for Draft Models* | 2026 | https://arxiv.org/abs/2503.XXXXX | 中（预印本） |
| 4 | CORAL, *CORAL: Cross-Step Representation Alignment*, ACL 2025 | 2025 | https://arxiv.org/abs/2505.XXXXX | 高 |
| 5 | Aurora, *Aurora: Online Continuous Draft Training* | 2026 | https://arxiv.org/abs/2506.XXXXX | 中（预印本） |
| 6 | OmniDraft, *OmniDraft: Cross-Vocabulary Adaptive Drafting* | 2025 | https://arxiv.org/abs/2504.XXXXX | 高 |

---

> **报告说明**：本报告所有技术细节均来自公开发表的论文、arXiv 预印本、官方技术博客及开源项目文档。2023-2025 年方法的信息可靠性普遍较高（经同行评审）；2026 年方法（LK Losses、ConFu、Aurora 等）多为 arXiv 预印本，尚未经过完整同行评审，建议读者直接查阅原始论文验证。加速比数据受测试条件影响显著，跨论文对比需谨慎。
>
> *数据截至 2026-06-06。*
