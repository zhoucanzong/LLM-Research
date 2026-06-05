# DeepSeek 系列模型深度调研报告

> 调研范围：DeepSeek 自 2023 年 11 月初代 DeepSeek-LLM 至 2026 年 4 月 DeepSeek-V4 的全部公开模型，覆盖通用基座（V 系列）、推理（R1 系列）、视觉语言（VL/OCR）、代码（Coder）、数学与形式化证明（Math/Prover）、多模态统一生成（Janus）六大主线。
> 调研时点：2026-06。
> 本报告基于两份子报告整合：[纯文本基座模型](file:///Users/zhoucanzong/Documents/codes/research-report/papers/DeepSeek-research/deepseek-text-models.md) 与 [多模态与专业模型](file:///Users/zhoucanzong/Documents/codes/research-report/papers/DeepSeek-research/deepseek-multimodal-and-specialized.md)，并提炼跨方向的整合性洞察。

---

## 1. 概述与发展时间线

### 1.1 DeepSeek 整体定位：高效开源路线

DeepSeek（深度求索）由幻方量化（High-Flyer）于 2023 年 7 月孵化成立，自成立之初就在开源 LLM 赛道选择了一条与众不同的"工程纪律驱动"路线：

1. **不与闭源比 GPU 数量，而是比"每美元/每 token 的智能输出"**——把成本而非参数量作为优化目标，这一思想贯穿 V1 → V4。
2. **彻底开源（MIT/宽松商用许可）+ 完整技术报告**——从权重、tokenizer、训练日志细节到 RL 算法（GRPO）、训练框架（HAI-LLM）、并行算法（DualPipe），凡可公开者全部公开，使开源社区可以最大程度复现。
3. **架构差异化为先**——在 Qwen / LLaMA / Mistral 普遍跟随业界主流（Dense + GQA + RoPE + cosine LR）时，DeepSeek 在 V2 阶段就以 **MLA + DeepSeekMoE** 双核心做出与友商根本不同的架构选择，并在 V3/V3.2/V4 阶段持续以 FP8、MTP、Auxiliary-Loss-Free、DSA 等系统级优化拉开差距。
4. **专业方向多点开花、最终回流主线**——R1 推理、Coder 代码、Math 数学、Prover 证明、VL 视觉、Janus 生成均独立演进，但每条线的关键技术（GRPO、自验证、混合编码、动态切片）最终都被吸收回 V 系列主基座，形成"分叉—回流"的闭环开发模式。

与 Qwen 等同时期开源系列相比，DeepSeek 的差异化定位体现在：

| 维度 | DeepSeek 路线 | Qwen 路线（参考） |
| --- | --- | --- |
| 架构 | MLA + DeepSeekMoE 一以贯之；强调"低激活高总参" | 主线 Dense（Qwen2/2.5/3）+ MoE 旗舰，更接近业界主流 |
| 训练目标 | 极致成本（V3 \$5.576M 训完 671B/14.8T）+ MTP 训练信号增强 | 多尺寸覆盖优先，强调小模型可用性 |
| 推理模型 | 纯 RL 路线（R1-Zero 无 SFT 冷启动）+ 规则化奖励 | 偏 SFT + RL 组合 |
| 多模态生成 | 解耦视觉编码（Janus 系列） | Qwen-VL/Omni 单编码器+多模态对齐 |
| 长上下文 | DSA 稀疏注意力 + Token-wise 压缩，目标 1M | YaRN/双块注意力等渐进式扩展 |
| 开源策略 | 同步公开权重 + 技术报告 + 训练框架细节 | 同步开源权重，技术细节相对收敛 |

### 1.2 2024–2026 完整时间线

```text
─────────────────────────  2023  ─────────────────────────
2023-11  DeepSeek-LLM-7B/67B（首发，Dense；4K 上下文）
─────────────────────────  2024  ─────────────────────────
2024-01-05  DeepSeek-LLM (论文 arXiv:2401.02954) — 自研 scaling laws
2024-01-11  DeepSeekMoE (arXiv:2401.06066) — 细粒度专家分割 + 共享专家隔离
2024-01-25  DeepSeek-Coder（1.3B–33B Dense；2T tokens；仓库级 FIM）
2024-02-05  DeepSeekMath（7B；提出 GRPO，arXiv:2402.03300）
2024-03-08  DeepSeek-VL（1.3B/7B；混合视觉编码器）
2024-05-07  DeepSeek-V2（236B/21B MoE；首次引入 MLA + DeepSeekMoE；128K）
2024-05-17  V2-0517（IFEval 63.9 → 77.6）
2024-05-23  DeepSeek-Prover-V1（Lean 4 自动证明；miniF2F 50.0%）
2024-06-17  DeepSeek-Coder-V2（236B/21B MoE；338 种语言；128K）
2024-06-28  V2-0628（HumanEval 84.76）
2024-08-15  DeepSeek-Prover-V1.5（RLPAF + RMaxTS；miniF2F 63.5%）
2024-09-05  DeepSeek-V2.5（V2-Chat ⊕ Coder-V2 合并）
2024-10-17  Janus（1.3B；解耦视觉编码）
2024-11    JanusFlow；R1-Lite-Preview（线上对标 o1-preview）
2024-12-10  V2.5-1210（MATH-500 82.8）
2024-12-13  DeepSeek-VL2（Tiny/Small/Base；动态切片 + DeepSeekMoE + MLA）
2024-12-26  DeepSeek-V3（671B/37B；FP8 + MTP + ALF；14.8T；\$5.576M）
─────────────────────────  2025  ─────────────────────────
2025-01-20  DeepSeek-R1-Zero / R1（arXiv:2501.12948）— 纯 RL 涌现推理
2025-01-22  R1-Distill × 6（Qwen 1.5/7/14/32B + Llama 8/70B）
2025-01-27  Janus-Pro（1B/7B）— 现象级开源生成模型
2025-03-24  DeepSeek-V3-0324（MMLU-Pro 81.2，AIME 59.4）
2025-04-30  DeepSeek-Prover-V2（7B + 671B；子目标分解 + cold-start RL；miniF2F 88.9%）
2025-05-28  DeepSeek-R1-0528（AIME 2025 70 → 87.5）+ R1-0528-Qwen3-8B
2025-08-21  DeepSeek-V3.1（首个 Hybrid Think / Non-Think 单一 checkpoint）
2025-09-22  V3.1-Terminus（中英混杂修复）
2025-09-29  V3.2-Exp（首次引入 DeepSeek Sparse Attention DSA）
2025-10-20  DeepSeek-OCR（Contexts Optical Compression；96%+ 精度）
2025-11-27  DeepSeekMath-V2（自验证 + 自精炼 + meta-verifier；IMO 金牌级）
2025-12-01  DeepSeek-V3.2 / V3.2-Speciale（GRPO 升级 + 长 CoT；IMO/IOI 双金）
2025-12-31  mHC（研究）— Manifold-Constrained Hyper-Connections
─────────────────────────  2026  ─────────────────────────
2026-01-04  R1 论文 v2（Nature 645:633-638 正式发表）
2026-04-24  DeepSeek-V4-Pro / V4-Flash（1.6T/49B + 284B/13B；Token-wise 压缩 + DSA；1M 上下文默认）
```

---

## 2. 基座语言模型演进

DeepSeek 基座主线呈现一条**单向收敛**的路线：从 Dense 到 MoE，从 BF16 到 FP8，从 4K 到 1M 上下文，每一代都明确"加上一个能省成本/提质量的核心机制"。

### 2.1 DeepSeek-LLM (2024-01)：Dense 起跑点

- 7B / 67B 双档 Dense Transformer（67B 上使用 GQA），BBPE 100K 词表（**这一 tokenizer 自此从未变过**），4K 上下文，2T tokens 预训练。
- 唯一的"原创武器"是**自研 scaling laws**：基于 non-embedding FLOPs/token (M) 的最优分配公式，纠正了 Chinchilla 的部分系数；67B Chat 直接超越 LLaMA-2 70B 与 GPT-3.5。
- 后训练 = SFT + DPO，对齐路线相对保守。

### 2.2 DeepSeek-V2 (2024-05)：架构革命的起点

V2 是 DeepSeek 历史上**最关键的一次架构跃迁**，一次性在同一篇论文里把后续两年的核心架构骨架确立下来：

- **236B 总 / 21B 激活**（V2-Lite 15.7B/2.4B），上下文一举跳到 **128K**（YaRN 扩展）。
- **首次引入 MLA**（攻 KV cache）+ **DeepSeekMoE FFN**（160 routed + 2 shared，激活 6 routed）+ **三层辅助损失负载均衡**（ExpBal/DevBal/CommBal）+ device-limited routing。
- 训练栈：BF16 + 16-way zero-bubble 流水线 + 8-way 专家并行 + ZeRO-1（不使用 tensor 并行），运行在 H800 集群，HAI-LLM 自研框架。
- 8.1T tokens 预训练；后训练首次引入 **GRPO**（来自同年 2 月的 DeepSeekMath）。
- **成本断崖**：每训练 1T tokens，DeepSeek 67B 需 300.6K H800·h，V2 仅需 172.8K，**节省 42.5%**；推理 FP8 + 6-bit KV 量化使最大吞吐相对 67B 提升 **5.76×**。
- 后续小迭代：V2-0517（IFEval 大幅提升）、V2-0628（HumanEval/MATH/BBH 全面增强）、**V2.5（V2-Chat 与 Coder-V2 合并为单一模型）**、V2.5-1210（MATH-500 进一步推到 82.8）。

### 2.3 DeepSeek-V3 (2024-12)：开源 LLM 的工程巅峰

V3 在 V2 架构骨架上叠加了三项系统级创新，把 671B 的训练成本压到 **\$5.576M**——这是 2024 年最具冲击力的开源工程纪录：

- **671B 总 / 37B 激活**（256 routed + 1 shared，激活 8 routed），61 层中前几层为 dense FFN。
- **Auxiliary-Loss-Free 负载均衡**：用不参与梯度的 per-expert bias 替代主要辅助损失，仅保留极小权重的 sequence-wise 保险损失，**负载均衡与主任务质量同时改进**。
- **Multi-Token Prediction (MTP)**：在主 next-token loss 之外多预测 1 个 token，训练信号更密 + 推理时作为 speculative decoding 的 draft module，**第二 token 接受率 85–90%，端到端 TPS ×1.8**。
- **FP8 混合精度训练**：首个在 671B 量级跑通的开源 FP8 端到端训练；统一 E4M3 + tile-wise/block-wise 量化 + FP32 高精度累加；显存减半 + GEMM 算力翻倍，**14.8T tokens 全程零 loss spike、零回滚**。
- **DualPipe** 流水线 + 跨节点 all-to-all kernel：MoE 通信开销几乎被完全 overlap。
- 14.8T tokens 预训练，后训练 SFT + RL 并蒸馏部分 R1 知识。
- V3-0324（2025-03）继续推升：MMLU-Pro 75.9 → 81.2，AIME 39.6 → 59.4，LiveCodeBench 39.2 → 49.2。

### 2.4 DeepSeek-V3.1 (2025-08)：从双轨到混合推理

- 架构与 V3 完全一致，**改变的是后训练**：用同一个 checkpoint，通过 chat template 切换 Think / Non-Think 双模式（`deepseek-chat` = 非思考，`deepseek-reasoner` = 思考），CoT 长度比 R1-0528 压缩 20–50%。
- Agent 能力大幅增强：SWE-bench Verified 66.0、SWE-bench Multilingual 54.5、Terminal-bench 31.3。
- V3.1-Terminus（2025-09-22）小修中英混杂、异常字符，并成为 V3.2 的训练 base。

### 2.5 DeepSeek-V3.2 (2025-09 → 2025-12)：稀疏注意力时代

- **V3.2-Exp（2025-09-29）**：在 V3.1-Terminus 之上做 continued training，首次引入 **DeepSeek Sparse Attention (DSA)** —— lightning indexer + token selector（top-k≈2048），attention 复杂度 $O(L^2) \to O(Lk)$，长上下文成本大幅下降。
- **V3.2（2025-12-01，arXiv:2512.02556）**：架构与 Exp 相同，但训练流程升级——可扩展 RL 框架（性能与 GPT-5 相当）、大规模 Agentic Task 合成 pipeline、GRPO 升级（按域调权 KL、修正 KL 估计、off-policy 序列掩码、MoE rollout 路由保持等）、可验证 / 不可验证域分别采用规则奖励 / 生成式奖励 + DeepSeekMath V2 数据。
- **V3.2-Speciale**：仅推理数据 RL、放宽长度惩罚的"扩展思考"变体，IMO 2025 与 IOI 2025 双金。

### 2.6 DeepSeek-V4 (2026-04)：百万上下文成为默认

- **双模型策略**：

  | 子模型 | 总参 / 激活 | 定位 |
  | --- | --- | --- |
  | **V4-Pro** | 1.6T / 49B | 旗舰；对标 GPT/Gemini/Claude；开源 Agentic Coding SOTA |
  | **V4-Flash** | 284B / 13B | 高性价比；推理能力接近 Pro |

- **新型注意力**：*Token-wise 压缩 + DSA*——在 DSA 之上再叠一层 token-wise 隐空间压缩；**1M 上下文成为所有官方服务的默认上限**。
- 三档思考强度（Non-think / Think High / Think Max），与 Claude Code、OpenClaw、OpenCode 等 agent 生态无缝对接。
- 自 2026-07-24 起 `deepseek-chat`/`deepseek-reasoner` 旧名将退役，分别路由至 V4-Flash 的非思考 / 思考模式。

### 2.7 跨代架构对比表

| 维度 | V1 (2024-01) | V2 (2024-05) | V3 / V3.1 (2024-12 / 2025-08) | V3.2 (2025-12) | V4 (2026-04) |
| --- | --- | --- | --- | --- | --- |
| 模型类型 | Dense | MoE | MoE | MoE + 稀疏注意力 | MoE + 稀疏注意力 |
| 总参 / 激活 | 7B / 67B (Dense) | 236B / 21B（Lite 15.7B/2.4B） | 671B / 37B | 685B / ~37B | Pro 1.6T/49B；Flash 284B/13B |
| 注意力 | MHA / GQA | **MLA** | MLA | MLA + **DSA**（lightning indexer + top-k≈2048） | **Token-wise 压缩 + DSA** |
| FFN | SwiGLU Dense | DeepSeekMoE（160 routed + 2 shared，激活 6） | DeepSeekMoE（256 routed + 1 shared，激活 8） | 同 V3 | 升级版 DeepSeekMoE |
| 负载均衡 | — | ExpBal+DevBal+CommBal 三辅助损失 + device-limited | **Auxiliary-Loss-Free（per-expert bias）** + 极小 seq-wise loss | 沿用 V3 + DSA 友好路由 | 同 V3.x |
| 训练目标 | next-token | next-token | **+ Multi-Token Prediction (D=1)** | + MTP | + MTP（推断） |
| 训练精度 | BF16 | BF16 | **FP8 (E4M3) + 高精度累加 + tile/block 量化** | FP8 | FP8（推断） |
| 流水/通信 | 标准 ZeRO + PP | 16-way zero-bubble PP + 8-way EP + ZeRO-1 | **DualPipe** + 跨节点 all-to-all | 同 V3 | 同 V3.x |
| 上下文 | 4K | 128K（YaRN） | 128K | 128K | **1M 默认** |
| 预训练 tokens | 2T | 8.1T | 14.8T | 14.8T 之上继续训练 | 未公开 |
| 训练成本 (主算力) | — | 每 1T tokens 172.8K H800·h | **2.788M H800·h ≈ \$5.576M** | 未单独披露 | 未单独披露 |
| 后训练 | SFT + DPO | SFT + GRPO | SFT + RL（含 R1 蒸馏） | RLVR + 生成式 reward + GRPO 升级 | RLVR + Agentic 任务合成 |
| 推理形态 | Base + Chat | Base + Chat | Base + Chat（V3.1 起 hybrid） | hybrid + Speciale 扩展思考 | Pro/Flash + 三档思考 |

---

## 3. 核心技术创新深度解析

DeepSeek 的差异化几乎完全建立在六项核心技术上——**MLA、DeepSeekMoE、FP8 训练、MTP、ALF、DSA**——其中 MLA 与 DeepSeekMoE 是最具标识性的"DeepSeek 烙印"。

### 3.1 MLA（Multi-head Latent Attention）

**问题背景**：对一个 N 层、$n_h$ 头、每头 $d_h$ 维、序列长度 L 的 MHA，KV cache 大小为 $2 N n_h d_h L$ bytes。在 100B+ 模型 + 128K 上下文场景下，单 batch KV cache 通常以**百 GB**计，是 prefill/decoding 的主导开销。

**历史方案**：

| 方案 | 思路 | 缺点 |
| --- | --- | --- |
| MHA | 每头独立 K/V | KV 巨大 |
| MQA (Shazeer 2019) | 所有头共享一组 K/V | KV 极小但损质量明显 |
| GQA (Ainslie 2023) | 头分 g 组，每组共享 K/V | g 上做权衡，超长上下文仍偏大 |

**MLA 的核心三步**（V2 论文 §2.1）：

1. **Joint low-rank compression**：用降投影 $W^{DKV}$ 把隐状态 $\mathbf{h}_t$ 压成潜向量 $\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t$（V2 中 $d_c = 4d_h$），K 与 V 共用一份压缩潜向量。
2. **延迟上投影 + Absorb 技巧**：$\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}$，$\mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}$；得益于矩阵结合律，$W^{UK}$ 在推理时可被吸收进 $W^Q$，从而**只需缓存 $\mathbf{c}_t^{KV}$**（每 token 仅一份、不分头）。
3. **解耦 RoPE 通道**：因 RoPE 与低秩压缩不可交换（合并矩阵会被 RoPE 阻断），单独维护一个共享小维度的 RoPE 路径 $\mathbf{k}_t^R$（维度 $d_h^R = d_h/2$），与 query 拼接为 $[q_t^C; q_t^R]$、key 拼接为 $[k_{t,i}^C; k_t^R]$，使位置信息独立维护、不破坏低秩 absorb 优化。

**KV cache 体积对比**（V2 论文 Table 1）：

| 注意力机制 | 每 token KV cache 元素数 | 能力 |
| --- | --- | --- |
| MHA | $2 n_h d_h l$ | Strong |
| GQA (g 组) | $2 n_g d_h l$ | Moderate |
| MQA | $2 d_h l$ | Weak |
| **MLA (DeepSeek-V2)** | $(d_c + d_h^R) l \approx \tfrac{9}{2} d_h l$ | **Stronger** |

MLA 的 KV cache 体积**约等于 GQA 仅 2.25 组**的水平，但效果**强于 MHA**——这是 V2 能把 128K 上下文做到推理可负担的关键。

**信息论视角**：

- GQA 是**结构性容量裁剪**：直接减少 K/V 头数，等价于让所有头共用同一 subspace，损失多视角能力。
- MLA 是**信息论级压缩**：压缩到 $d_c$ 维潜空间，但通过解耦 RoPE 与 absorb 技巧，**模型参数容量与计算能力并未削弱**——KV cache 减少来自"潜空间储存 + 推理时上投回原空间"，类似 LoRA 的 down-project / up-project 之于权重。这也是 V2 论文 Appendix D 中 MLA 在同 KV 体积下质量优于 MHA 的根因。

**演化轨迹**：

- **V2 → V3**：MLA 结构不变，但通过 FP8 + DualPipe 把训练效率推到极致；MLA 的 KV cache 进一步以 6-bit 量化部署。
- **V3 → V3.2**：MLA 之上叠加 DSA，注意力复杂度从 $O(L^2)$ 降到 $O(Lk)$；DSA 的 lightning indexer 直接复用 MLA 已压缩的 key latent，**避免重复缓存**——这是 MLA 与 DSA 协同设计的关键巧思。
- **V3.2 → V4**：在 DSA 之上再引入 token-wise 压缩，把 1M 默认上下文做到可负担。
- **跨方向复用**：DeepSeek-VL2 直接搬用 MLA 来支撑高分辨率长视觉序列；R1 系列继承 V3-Base 的 MLA 处理长 CoT。

### 3.2 DeepSeekMoE 设计哲学与迭代

**标准 MoE 的两个痛点**（GShard / Switch Transformer 时代）：

- **专家专业化不足**：经典做法把 FFN 切成几十到一百多个专家，每个专家仍然学到大量"通用知识"，专家间冗余高。
- **routed 专家承担公共知识** 导致负载更难均衡，且容易出现"路由塌缩"。

**两大设计原则**（DeepSeekMoE 论文 arXiv:2401.06066）：

1. **Fine-grained Expert Segmentation（细粒度专家分割）**：保持 FLOPs 不变，把每个常规专家切成 m 等份的小专家，同时把激活的专家数量也乘以 m。在不增加每 token 计算量的前提下，**专家组合空间从 $\binom{N}{K}$ 扩张到 $\binom{mN}{mK}$**，专业化粒度大幅提升。
2. **Shared Experts Isolation（共享专家隔离）**：拿出极少量（V2 中 2 个、V3 中 1 个）"共享专家"对所有 token 始终激活，承担"公共语言/通用知识"，让 routed experts 把容量留给真正需要专业化的方向。

**V2 阶段：多重辅助损失 + device-limited routing**

- **ExpBal**：基于路由频率 $f_i$ 与平均概率 $P_i$ 的乘积惩罚不均衡。
- **DevBal**：把专家划分为 D 组（每组一台设备），按设备维度计算同样形式的损失。
- **CommBal**：对接收侧通信量做均衡，配合 device-limited routing（每 token 只能落到 ≤ M 台设备）。
- 还引入 **Token-Dropping**：超过设备容量的 token 直接 drop。

**V3 阶段：Auxiliary-Loss-Free Load Balancing**

V2 的辅助损失虽然有效，但**会损害模型主任务性能**——损失函数中加入"平均化"压力会与"语义专精"目标冲突。V3 的解法：

- 用 **per-expert bias term** 替代主要辅助损失：$\tilde{s}_{i,t} = s_{i,t} + b_i$，bias 通过移动平均更新，**不参与梯度反传**——既能调度负载又不污染主任务损失。
- 仅保留极小权重的 **sequence-wise auxiliary loss** 作为防止极端不均衡的保险。
- 这是 V3 同时取得"负载更均衡 + 主任务效果更好"的关键之一。

**路由与并行：从 device-limited 到节点感知**

- V2：device-limited routing（≤ M 设备）。
- V3：node-limited routing + cross-node all-to-all 优化（IB + NVLink 双层 kernel），**MoE 通信开销几乎被 DualPipe 完全 overlap**。
- 推理：MLA 的小 KV + DeepSeekMoE 的稀疏激活，使 671B/37B 这种"低激活高总参"模型可以在远小于其总参的显存下高效部署。

### 3.3 FP8 训练与低成本策略

V3 是**首个在 671B 量级 LLM 上跑通 FP8 端到端训练**的开源模型。技术细节：

- 大部分 GEMM（Wgrad/Dgrad/Fwd）使用 **FP8 E4M3**；BF16/FP32 仅保留在 norm、loss、master weight、optimizer state、attention softmax 等敏感部分。
- **Tile-wise / Block-wise fine-grained 量化**：对激活按 1×128 tile 量化、对权重按 128×128 block 量化，避免 tensor-wise scaling 在 outlier 下溢出。
- **高精度累加**：FP8 GEMM 的部分和按周期升级到 FP32 累加（修补 Hopper 架构 FP8 累加器精度不足的问题）。
- **统一 E4M3**：与 NVIDIA 默认配方（forward E4M3 + backward E5M2）不同，V3 全用 E4M3，简化框架并配合 fine-grained 量化保证数值稳定。
- 收益：相比 BF16 训练**显存减半 + GEMM 算力翻倍**，没有任何 loss spike 或回滚。

**配套低成本策略**：

- **DualPipe**：自研流水线算法，让前向和反向同时双向流动，**几乎完全重叠 all-to-all 通信与计算**。
- **HAI-LLM 自研框架**：16-way zero-bubble 流水线 + 8-way 专家并行 + ZeRO-1，**未使用 tensor 并行**（因激活参数少 + 部分算子重计算），定制 CUDA kernel 与改造版 FlashAttention-2。
- **训练成本拆解**（V3 论文摘要原话）：

  | 阶段 | GPU 小时 (H800) | 折算 (\$2/h) |
  | --- | --- | --- |
  | 预训练 | 2.664M | \$5.328M |
  | 上下文扩展 | 0.119M | \$0.238M |
  | 后训练 | 5K | \$10K |
  | **总计** | **2.788M** | **≈ \$5.576M** |

- 该成本仅含主训练 GPU 算力，**不包含数据、消融实验、人力等成本**——这一点 DeepSeek 团队也在多个采访中澄清过，但即便如此，与同级闭源旗舰相比仍有数量级优势。

### 3.4 Multi-Token Prediction (MTP)

**机制**（V3 论文 §2.2）：

- 训练目标在主 next-token loss 之外，**额外预测下一 D 个 token**（V3 中 D=1，即除当前位置外再多预测 1 个 token）。
- 实现上是在主模型之上挂载一个**轻量级 MTP 模块**（一层 transformer + projection），共享主干 embedding/output head，在训练阶段只增加少量 FLOPs。

**两点收益**：

1. **训练信号更密**：主模型本身在所有标准基准上得到稳定提升（论文消融数据均为正向）。
2. **推理时作为投机解码 draft module**：论文报告**第二 token 接受率约 85–90%**，端到端 TPS 大约 ×1.8。

这是大型 MoE 上**首次将 MTP 工业级落地**——同期业界（Meta MTP 论文 2024）尚停留在小模型实验阶段。

### 3.5 Auxiliary-Loss-Free 负载均衡

详见 [3.2 节](#32-deepseekmoe-设计哲学与迭代) 的 V3 段落。其核心洞察是：

> 负载均衡是工程目标，不应通过损失函数让模型"妥协"。
> 正确的做法是把它放到优化器旁边——一个**不参与梯度的调度器**——用 bias 移动平均做闭环控制。

这个思想后来被多个开源 MoE 项目（DBRX、Qwen-MoE 等）借鉴。

### 3.6 DeepSeek Sparse Attention (DSA)

V3.2 首次引入，是 DeepSeek 把"长上下文成本"打下去的关键武器。

**两个组件**：

1. **Lightning Indexer**——为每个 query token 计算其对所有历史 token 的"相关度评分"。在 MLA 压缩空间中复用 keys（**避免重复存储**），并使用 ReLU 与 per-head 学习权重的简单形式：
   $$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}\, \mathrm{ReLU}(q_{t,j} \cdot k_s)$$
   其中 t 是当前 query 位置，s 是历史 token 位置（s<t），j 是 indexer head 索引；keys 直接从 MLA 已压缩并缓存的 latent 取出。

2. **Token Selector**——基于 indexer 分数选 **top-k**（DeepSeek 公开代码中 k=2048），构造稀疏 attention mask，只对被选中的历史 token 计算 attention。

**效果**：

- 注意力复杂度从 $O(L^2)$ 降为 **$O(L \cdot k)$**（k≪L），对长上下文的训练与推理成本大幅下降。
- 设计目标不是"超过 V3.1-Terminus"，而是在长上下文上**几乎不损失质量**的前提下取得效率优势。

**V4 升级**：在 DSA 之上叠加 **token-wise 压缩**——对 token 序列在隐空间做压缩（细节官方未完全披露），使 1M 上下文成为所有官方服务的默认上限。

---

## 4. 推理模型演进

### 4.1 R1-Lite-Preview (2024-11)：探路

发布形式为 chat.deepseek.com 上线的预览版（未开源 weights，未发表正式论文），是 DeepSeek 第一次正面回应 OpenAI 在 2024 年 9 月发布的 o1-preview。在 AIME、MATH-500 上首次接近 o1-preview 水平，验证了"长思维链 + 强化学习"路径可行。

### 4.2 R1-Zero (2025-01)：纯 RL 涌现推理

**论文**：DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948（Nature 645:633-638, 2025）。

- **基座**：DeepSeek-V3-Base（671B / 37B 激活）。
- **训练范式**：完全跳过监督微调（SFT）冷启动，**直接对基座做大规模 RL**。这是首个公开证明**无需任何人类标注推理轨迹也能让 LLM 涌现自反思、验证、回溯、动态策略切换等高阶推理行为**的开源模型。
- **算法**：使用 **GRPO**（Group Relative Policy Optimization，源自 DeepSeekMath）：
  - 去掉 PPO 的 critic value model，改用**组内相对优势**——对同一 prompt 采样 G 条响应，用组内归一化得分作 advantage。
  - 显存与算力开销显著低于 PPO，特别适合长 CoT。
- **奖励设计**：完全规则化，避免奖励模型 hacking——
  1. *Accuracy reward*：数学题答案匹配、代码题编译/单测通过；
  2. *Format reward*：必须以 `<think>...</think><answer>...</answer>` 包裹推理过程。
- **观察到的 "Aha Moment"**：模型在训练中段自发出现"等等、让我重新检查……"这类自我修正行为，思考长度随训练步数单调上升。
- **缺陷**：可读性差、中英混杂、有时输出冗长重复。

### 4.3 R1 (2025-01)：四阶段管线对标 o1-1217

为缓解 R1-Zero 的可读性问题，DeepSeek 设计了四阶段训练管线：

1. **冷启动 SFT**：用数千条高质量长 CoT 数据（部分由 R1-Zero 生成后人工修订）对 V3-Base 做轻量 SFT，规范输出格式。
2. **面向推理的 RL**：与 R1-Zero 同样的 GRPO + 规则奖励，并加入语言一致性奖励避免中英混杂。
3. **拒绝采样 + SFT**：用阶段 2 收敛模型生成 ~600K 推理数据 + 200K 通用数据（写作、知识问答、自我认知）一起 SFT，让模型在保留推理能力的同时回归通用对齐。
4. **全场景 RL**：最后再做一轮 RL，目标兼顾 helpfulness/harmlessness（用奖励模型）与 accuracy（用规则奖励），完成最终对齐。

- **训练成本**：根据 Nature 补充材料披露，R1 仅花费约 **US\$294K**（不含 V3-Base 约 600 万美元的预训练）。
- **性能**：AIME 2024 pass@1 ≈ 79.8%，MATH-500 97.3%，Codeforces 2029 ELO（96.3 percentile），与 OpenAI o1-1217 持平或超越。
- **协议**：MIT License；首个面向 LLM 推理的"完全开源 + 完全可商用"前沿模型。

### 4.4 R1-Distill：从 671B 装进小模型

DeepSeek 用 R1 生成 ~80 万条推理样本（≈60 万推理 + 20 万通用），对 6 个开源 dense 基座做单纯 SFT（**故意不做 RL**，旨在测试"知识注入是否足够"）：

| 蒸馏模型 | 基座 | AIME 24 pass@1 | MATH-500 | LiveCodeBench |
| --- | --- | --- | --- | --- |
| R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 28.9 | 83.9 | 16.9 |
| R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 55.5 | 92.8 | 37.6 |
| R1-Distill-Llama-8B | Llama-3.1-8B | 50.4 | 89.1 | 39.6 |
| R1-Distill-Qwen-14B | Qwen2.5-14B | 69.7 | 93.9 | 53.1 |
| R1-Distill-Qwen-32B | Qwen2.5-32B | **72.6** | 94.3 | 57.2 |
| R1-Distill-Llama-70B | Llama-3.3-70B-Instruct | 70.0 | 94.5 | **57.5** |

**关键结论**：

- **32B 蒸馏版在多项数学/编码基准上超过 OpenAI o1-mini**，是当时社区最强的"消费级硬件可部署"推理模型。
- 论文做了对照实验：直接把 R1 的 RL 管线复用到 Qwen-32B-Base，得到的模型反而显著弱于"用 R1 蒸馏的 Qwen-32B"。说明**对小模型而言，从大模型蒸馏 CoT 比自己跑 RL 更高效**。

### 4.5 R1-0528 (2025-05)：思考预算翻倍

- 体量从 671B 微调到 685B（新增 token & 路由专家）。
- **思考预算翻倍**：AIME 题目平均思考 token 12K → 23K，AIME 2025 准确率 70% → 87.5%。
- 幻觉降低 45–50%（改写、摘要、阅读理解任务）。
- 新增 **JSON 输出**与**函数调用**支持；支持 system prompt；不再需要手动塞入 `<think>\n` 触发推理。
- 前端代码 / vibe-coding 显著增强。
- 同步释出 `DeepSeek-R1-0528-Qwen3-8B` 蒸馏版。

### 4.6 GRPO 训练范式深度解析

GRPO 是 DeepSeek 最具影响力的"算法导出"，最早提出于 DeepSeekMath（2024-02），在 R1 中验证、在 V3.2 中升级，已成为开源 RL 训练事实上的工业标准之一。其相对 PPO 的核心改动：

| 维度 | PPO | GRPO |
| --- | --- | --- |
| Critic / Value Model | 需要，与 policy 同尺寸 | **去掉，节省一半显存** |
| Advantage 估计 | GAE（基于 value） | **组内相对优势**：对同 prompt 采样 G 条 rollout，按组内得分归一化 |
| 算力开销 | 高 | 显著降低，适合长 CoT |
| 是否兼容规则化奖励 | 兼容 | **天然兼容**（V3.2 进一步引入按域调权 KL） |

V3.2 阶段的 GRPO 升级版：保留 KL（按域调权，数学域近 0）、修正 KL 估计、off-policy 序列掩码、MoE rollout 路由保持、top-p/top-k 采样掩码保持、保留原 GRPO normalization。

### 4.7 "无 SFT 冷启动"路线的意义

R1-Zero 的纯 RL 路线在 LLM 史上有特殊地位：

- **挑战了"SFT 是 RLHF 必备前置"的共识**。GRPO + 规则奖励 + 长 CoT 模板，足以让基座自发涌现推理能力。
- **降低了高质量推理数据的依赖**——这对竞赛级数学/代码问题尤为关键，因为高质量人类标注 CoT 极度稀缺。
- **奖励 hacking 防御**：完全规则化奖励 + 格式奖励，避免了"奖励模型 hack 奖励模型"的常见失败模式。
- **后续路线分叉**：R1（双轨）→ V3.1（混合推理）→ V4（三档思考强度）显示 DeepSeek 已在尝试把推理能力**收编进通用基座**而非永远保持双模型，这是 2025 下半年开源社区的主流方向。

---

## 5. 视觉语言模型演进

### 5.1 DeepSeek-VL (2024-03)：混合视觉编码器

第一代 VL 模型（arXiv:2403.05525），覆盖 1.3B / 7B 两档 dense LLM 基座。三大设计支柱：

1. **数据 (Data)**：覆盖网页截图、PDF、OCR、图表、知识类内容，并基于真实用户场景构建任务分类法（taxonomy）和指令微调集，强化"实战能力"。
2. **混合视觉编码器 (Hybrid Vision Encoder)**：
   - 高分辨率分支：基于 SAM-B（图像分辨率高达 1024×1024）；
   - 低分辨率分支：基于 SigLIP-L（语义浓缩）；
   - 两路特征经独立 MLP 投影后在通道维拼接，再过一个 MLP 进入 LLM。这一设计兼顾文档级细节（小字、表格）和宏观语义。
3. **三阶段训练 + 视觉-语言均衡**：
   - Stage 1：仅训练 vision-language adaptor；
   - Stage 2：联合预训练，从一开始就把 LLM 一起训，并引入精心设计的视觉/语言 token 比例调度，避免视觉训练侵蚀 LLM 能力；
   - Stage 3：监督微调 + 偏好对齐。

### 5.2 DeepSeek-VL2 (2024-12)：动态切片 + DeepSeekMoE + MLA

VL2（arXiv:2412.10302）围绕"高分辨率 + MoE 高吞吐"做两处重大升级：

**视觉侧：动态切片 (Dynamic Tiling)**

- 用一个固定基础分辨率的视觉编码器（SigLIP-SO400M-384）处理任意宽高比、任意分辨率图像。
- 输入图被切片成若干 384×384 patch + 一张 thumbnail，token 序列拼接后送入 LLM。
- 相比 VL 的"双分支固定 1024×1024"，VL2 能处理更长的纵向截图、4K 大图、文档页面，对 OCR/Doc/Chart/Table 任务尤其友好。

**语言侧：DeepSeekMoE + MLA**

- 改用 DeepSeek V2/V3 同款 **DeepSeekMoE** 架构（共享专家 + 细粒度专家）。
- 引入 **MLA**：把 KV cache 压缩到低秩 latent 向量，长上下文推理显存降一个数量级。
- 这是 DeepSeek 主基座技术**首次回流到 VL 子方向**。

**三个尺寸（按激活参数）**：

| 模型 | Activated | Total | 备注 |
| --- | --- | --- | --- |
| DeepSeek-VL2-Tiny | 1.0B | 3.4B | 边缘部署 |
| DeepSeek-VL2-Small | 2.8B | 16.1B | 平衡型 |
| DeepSeek-VL2 | 4.5B | 27.5B | 旗舰 |

**能力**：VQA、OCR、文档/表格/图表理解、视觉定位（grounding）。在同等激活参数下击败大量 dense 与 MoE 开源模型。

### 5.3 DeepSeek-OCR (2025-10)：上下文光学压缩

- 一个 3B VLM（arXiv:2510.18234），主线**不是"OCR 工具"**而是 **Contexts Optical Compression** 的研究：把整页文档压成极少量视觉 token（远少于直接 OCR 后的文本 token），仍保持 96%+ 解码精度。
- 隐含结论：未来可以用"页面图像 → 视觉 token → LLM"的方式**显著压缩**长文本上下文，是 LLM 上下文扩展的潜在新方向，与 V3.2 DSA 形成互补——一个是"在 token 层面稀疏选择"，一个是"在感知层面光学压缩"。

---

## 6. 代码与数学模型演进

### 6.1 DeepSeek-Coder 系列

**DeepSeek-Coder (2024-01-25, arXiv:2401.14196)**：
- 1.3B / 5.7B / 6.7B / 33B 共四档 dense，全部从零训练在 2T token 上（87% 代码 + 13% 自然语言，含中英）。
- **仓库级数据组织**：使用拓扑排序在仓库范围内拼接多文件，让模型理解跨文件依赖；上下文 16K。
- **Fill-in-the-Middle (FIM)**：训练时按比例随机执行"前缀-后缀-中段"重排，使模型同时具备 left-to-right 生成与中段补全能力，是 IDE 内联补全的核心技术。
- **两阶段训练**：通用代码预训练 + 数学/项目级再训练；继之以 instruction tuning 得到 `DeepSeek-Coder-Instruct`。
- 33B 版本是当时最强开源代码模型，HumanEval/MBPP/DS-1000 上超越 CodeLlama，并击败 Codex / GPT-3.5。

**DeepSeek-Coder-V2 (2024-06, arXiv:2406.11931)**：
- 在 **DeepSeek-V2-Base**（MoE）上做"代码继续预训练"，得到 Lite（16B/2.4B 激活）与 V2（236B/21B 激活）两档。
- **支持语言数 86 → 338**；**上下文 16K → 128K**；继续预训练数据 6T token，FIM 策略保留并优化。
- 后训练阶段加入 *DPO* + *GRPO*（数学/代码 RL，与 DeepSeekMath 同源）。
- HumanEval 90.2、MBPP+ 76.2、LiveCodeBench、SWE-Bench 等基准上**首次让开源 MoE 在代码智能上对标 GPT-4-Turbo / Claude-3-Opus / Gemini-1.5-Pro**。

**收编为主线**：2024-09 V2.5 把 V2-Chat 与 Coder-V2 合并升级，标志着"通用 + 代码"分支再度合流——**独立 Coder 系列暂未见 V3 形态**，因为 V3/V3.1/V3.2 在 LiveCodeBench、SWE-Bench Verified、Codeforces 上持续推进，事实上成为 DeepSeek 当前的"代码主战场"。这是 DeepSeek "专业方向最终回流主线"模式的典型案例。

### 6.2 DeepSeekMath 系列

**DeepSeekMath (2024-02-05, arXiv:2402.03300)**：
- 7B dense 模型，从 DeepSeek-Coder-Base 7B 出发继续预训练于 120B 数学 token（高质量爬取 Common Crawl + 自训练分类器去噪）。
- 在 MATH 上以 7B 体量实现 **51.7%**，逼近 GPT-4 (52%) 与 Gemini-Ultra。
- **最大贡献：提出 GRPO**（详见 §4.6）——这篇论文事实上是**后来 R1 训练算法的"前传"**。

**DeepSeekMath-V2 (2025-11-27, arXiv:2511.22570)**：
- 基于 V3.2-Exp-Base 的数学专用模型。
- **目标转向"自验证"（self-verifiable）数学推理**：模型不仅要给答案，还要生成可由模型自己复核的*证明*；训练目标从"答案匹配"转为"证明合格"，从根本上抑制"对答案胡编"的奖励作弊。
- 两个新方向：
  1. *Process-level verification*：把奖励信号从最终答案搬到推理过程中的每个步骤；
  2. *Theorem-proving 与一般数学推理融合*：吸收 Prover 系列的形式化证明能力。
- **三 LLM 链路**：训练独立的 LLM verifier（LLM 2）对 generator（LLM 1）输出打分，再训练 meta-verifier（LLM 3）评估 verifier；rubric 评分为 1/0.5/0。
- **Self-Refinement**：训练好的 generator 学会按 rubric 对自己的输出迭代修正（最多 8 轮，仍未饱和）。
- 在 IMO 2025、CMO 2024 上取得**金牌级**得分；其能力被并入 V3.2-Speciale。

### 6.3 DeepSeek-Prover 系列：Lean 4 形式化定理证明

| 版本 | 时间 | arXiv | 关键贡献 |
| --- | --- | --- | --- |
| **Prover-V1** | 2024-05-23 | 2405.14333 | 首版：从高中/竞赛题自动 *autoformalize* 为 Lean 4 statement，再用模型生成证明，用 Lean 4 验证作为 ground truth；筛选出 8M 高质量合成证明对 7B 模型做 SFT，miniF2F-test 达 50.0% |
| **Prover-V1.5** | 2024-08-15 | 2408.08152 | 继续预训练 + RLPAF（用 Lean proof-assistant 反馈做 RL）+ **RMaxTS**（Monte-Carlo Tree Search 引入 intrinsic reward 鼓励多样路径）。miniF2F 63.5%，ProofNet 25.3% |
| **Prover-V2** | 2025-04-30 | 2504.21801 | 671B（基于 V3）+ **子目标分解**：先用大模型把非形式化证明拆成 subgoal，再让 7B prover 逐个完成、最后回拼。引入 *cold-start RL*：从一批 high-quality 证明开始，用 GRPO 微调；miniF2F **88.9%**，攻克 PutnamBench 相当一部分题；同时发布 7B 与 671B 双尺寸 |

形式化证明系列与 DeepSeekMath-V2 的"自验证"思想高度一致，可视为 DeepSeek 在"reasoning 可验证化"路径上的两个互补支柱。

---

## 7. 多模态生成（Janus）

### 7.1 解耦视觉编码的设计理念

之前的"Chameleon 路线"用**单一**视觉 tokenizer 同时承担"理解"（需高语义抽象）与"生成"（需细粒度像素级 token），两者目标冲突，导致理解性能下滑。

**Janus 的解法**（arXiv:2410.13848，CVPR 2025）：

| 任务 | 视觉编码器 | 输出空间 |
| --- | --- | --- |
| 多模态理解 | SigLIP-L（连续语义特征） | 投影到 LLM token 空间 |
| 图像生成 | VQ tokenizer（离散像素 token） | 自回归生成离散 image token |

- 两路**分别**编码，但**共享同一个 Transformer 主干**做下游建模。架构因此既统一又解耦。
- 训练分三阶段：adaptor 训练 → 统一预训练 → 指令微调。
- 在 GenEval、DPG-Bench 等图像生成基准与 MMBench 等理解基准上，**同时**优于此前的 unified models（Chameleon、Show-o）。
- 1.3B 参数（DeepSeek-LLM-1.3B 基座），首次给"统一理解 + 生成"模型一个清晰的架构指引。

### 7.2 JanusFlow (2024-11)：流匹配生成

把生成路径换成 **Rectified Flow**（连续流匹配），抛弃 VQ tokenizer，进一步提升生成质量与采样效率，仍延续解耦视觉编码思路。

### 7.3 Janus-Pro (2025-01)：现象级开源生成

- 1B 与 7B 双尺寸，相对 Janus 三处升级：
  1. **优化训练策略**：调整三阶段时长、视觉 token 比例；
  2. **扩展数据**：理解侧多模态 SFT 数据扩到 ~9000 万样本；生成侧加入 7200 万合成审美数据；
  3. **模型放大**至 7B。
- 在 GenEval、DPG-Bench 上**一度超过 SDXL、DALL·E 3**，在春节期间引爆了 Hugging Face 趋势榜，是 2025 年初的现象级开源生成模型。
- 与 R1 在同一周发布，构成 DeepSeek **2025 年初的"双爆点"**。

---

## 8. 跨代技术演进总结

### 8.1 核心设计哲学

**一、效率优先（Efficiency-First）**

DeepSeek 几乎所有架构与训练决策都可以归约为一个目标函数：**在同等智能水平下，把训练与推理成本系统性降低一个数量级**。

- 架构层：MLA 攻 KV cache、DeepSeekMoE 攻 FFN 计算、DSA 攻长上下文 attention、Token-wise 压缩攻 1M 上下文。
- 训练层：FP8（攻显存与 GEMM 算力）、DualPipe（攻流水线气泡）、ALF（攻"为均衡牺牲质量"）、MTP（攻训练信号密度）。
- 后训练层：GRPO（攻 critic 显存）、规则奖励（攻奖励 hacking）、自验证（攻"对答案胡编"）。
- 这种"全栈成本视角"使 DeepSeek 能用 \$5.576M 训出对标闭源旗舰的 671B MoE，是开源 LLM 工程史上最具纪律性的演化序列。

**二、开源开放（Open & Reproducible）**

- 权重 + 技术报告 + 训练框架 + RL 算法 + 后训练 pipeline 全部公开。
- 关键算法（GRPO、MLA、DeepSeekMoE、ALF、MTP、DSA）均有 arXiv 论文 + 完整数学推导 + 消融实验。
- 全部模型 MIT/宽松商用许可。
- 这种开放程度直接孕育了下游生态：R1-Distill 让 6 个开源 dense 基座（Qwen2.5/Llama-3.x）一夜之间获得 o1-mini 级推理能力，并被 OpenRLHF / TRL / verl 等 RL 框架收编。

**三、分叉—回流（Branch & Merge）**

DeepSeek 的演化模式具有清晰的"分叉—回流"特征：

```text
                 ┌─────────────── DeepSeek-V3-Base (671B MoE) ───────────────┐
                 │                                                            │
                 │         post-train ↓ (RLHF+SFT)        post-train ↓ (RL)   │
                 │                                                            │
        DeepSeek-V3-Chat                                  DeepSeek-R1-Zero / R1
              │                                                  │
   incorporated R1 RL ↓                              distill (SFT) ↓
              │                                                  │
   V3-0324 → V3.1 → V3.1-Terminus → V3.2-Exp → V3.2 → V4   R1-Distill × 6
                                          ⊕
                                  DeepSeekMath-V2
                                          ⊕
                                 (V3.2-Speciale 融合)
```

- Coder-V2 → V2.5 合并；R1 → V3.1 混合推理；DeepSeekMath-V2 → V3.2-Speciale；这些都是"专业线先证明可行，再回流主线"的典型路径。
- 这种模式既保留了专业方向的极致深度（如 Prover-V2 的 miniF2F 88.9%），又避免了主基座过度发散。

### 8.2 关键架构决策演变线索

| 决策维度 | 演化轨迹 | 触发动机 |
| --- | --- | --- |
| 注意力 | MHA/GQA → **MLA** → MLA + **DSA** → **Token-wise + DSA** | KV cache 瓶颈 → 长上下文 attention 复杂度 → 1M 上下文 |
| FFN | Dense SwiGLU → **DeepSeekMoE**（细粒度 + 共享专家） | "低激活高总参"性价比；专家专业化粒度 |
| 负载均衡 | 三层辅助损失 → **Auxiliary-Loss-Free** | 辅助损失伤害主任务质量 |
| 训练精度 | BF16 → **FP8 (E4M3)** + 高精度累加 | 显存与 GEMM 算力翻倍 |
| 训练目标 | next-token → **+ MTP** | 训练信号密度 + 推理 speculative decoding |
| 流水线 | 标准 PP → **DualPipe** | 跨节点 MoE all-to-all 通信开销 |
| 上下文 | 4K → 128K (YaRN) → **1M** (DSA + Token-wise) | 长文档、Agent、代码仓库 |
| 后训练 | SFT + DPO → **SFT + GRPO** → **RLVR + 规则奖励 + 自验证** | 奖励 hacking 防御；推理可验证化 |
| 推理形态 | Base + Chat 双模 → **R1 推理分叉** → **混合推理（Think/Non-Think）** → **三档思考强度** | 把推理收编进通用基座 |

### 8.3 与 Qwen 等其他系列的差异化定位

| 维度 | DeepSeek | Qwen（Alibaba） | LLaMA（Meta） |
| --- | --- | --- | --- |
| 架构主线 | MLA + DeepSeekMoE 一以贯之 | 主线 Dense + MoE 旗舰，跟随业界主流 | Dense + GQA |
| 模型尺寸覆盖 | 大尺寸为主（V2 起 ≥ 16B 总参） | **全尺寸**（0.5B–235B），强调小模型可用性 | 主要 7B–405B |
| 推理路线 | **纯 RL（R1-Zero）** + 规则奖励 | SFT + RL 组合 | 主要 Instruct 路线，推理依赖 prompt |
| 长上下文 | DSA 稀疏注意力 + Token-wise 压缩，目标 1M | YaRN/双块注意力等渐进式扩展 | RoPE 扩展 |
| 多模态生成 | **解耦视觉编码（Janus）** | 单编码器多模态对齐（Qwen-VL/Omni） | 较弱 |
| 训练框架 | 自研 HAI-LLM + DualPipe | 主要基于 Megatron-LM 改造 | Meta 内部框架 |
| 训练精度 | **率先工业级 FP8** | 主要 BF16 | BF16 |
| 开放程度 | 权重 + 技术报告 + 训练框架细节 | 权重开源，技术细节相对收敛 | 权重 + 论文 |

简言之，DeepSeek 的差异化定位可概括为："**在架构和训练栈上做架构革新，用同等智能下数量级低的成本回应闭源；以分叉-回流的方式让专业方向反哺通用基座。**"——这与 Qwen 的"全尺寸覆盖 + 跟随业界主流架构"路线、LLaMA 的"以参数规模和训练数据规模为主"的路线，形成了截然不同的工程文化。

---

## 9. 值得关注的关键创新点（按重要性排列）

1. **MLA（Multi-head Latent Attention）**——开源 LLM 中**唯一**对 KV cache 做信息论级压缩并在生产部署的方案。在同 KV 体积下质量优于 MHA、远优于 GQA/MQA；与 DSA、Token-wise 压缩天然协同（lightning indexer 直接复用 MLA latent）。**最具长期影响力的架构创新**——后续若有人挑战 GQA 的霸权，最可能的接班人就是 MLA 或其变体。

2. **DeepSeekMoE（细粒度专家分割 + 共享专家隔离 + Auxiliary-Loss-Free）**——重新定义了 MoE 设计哲学。"专家组合空间从 $\binom{N}{K}$ 扩张到 $\binom{mN}{mK}$"是一个被严重低估的洞察；ALF 让"负载均衡 + 主任务质量"首次能同时改进，已被 DBRX、Qwen-MoE 等借鉴。

3. **GRPO**——从 DeepSeekMath 提出，到 R1 验证为"无 SFT 也能涌现推理"的训练算法。已被 OpenRLHF、TRL、verl 等开源 RL 框架广泛收编，**事实上是 2025 年开源 RL 训练的工业标准**。

4. **R1-Zero 的纯 RL 路线**——首次公开证明无需任何人类标注推理轨迹也能让 LLM 涌现自反思、验证、回溯等高阶推理行为。"Aha Moment"是开源 LLM 史上的标志性现象。

5. **FP8 端到端训练（V3）**——首个在 671B 量级 LLM 上跑通的开源 FP8 训练，14.8T tokens **零 loss spike、零回滚**。统一 E4M3 + 1×128 tile / 128×128 block 量化 + FP32 高精度累加，是后续所有大型开源 MoE FP8 训练的参考实现。

6. **Multi-Token Prediction (MTP)**——大型 MoE 上首次将 MTP 工业级落地，训练信号增强 + 推理时 speculative decoding draft（接受率 85–90%，TPS ×1.8）。

7. **DSA (DeepSeek Sparse Attention)**——把长上下文 attention 复杂度从 $O(L^2)$ 降到 $O(Lk)$，并通过复用 MLA latent 避免重复缓存。是 V4 1M 默认上下文的基石。

8. **Janus 解耦视觉编码**——理解（语义连续特征）与生成（像素离散 token）分离编码、共享 Transformer 主干。在统一多模态生成路径上提供了清晰的架构指引，Janus-Pro 一度超越 SDXL/DALL·E 3。

9. **DualPipe 流水线 + 跨节点 all-to-all kernel**——让 H800 上的 MoE 通信开销几乎被完全 overlap，是 V3 \$5.576M 训练成本的工程基础。

10. **自验证 / 形式化奖励（DeepSeekMath-V2 + Prover-V2）**——把奖励从"答案匹配"搬到"证明合格"，把 generator / verifier / meta-verifier 三 LLM 链路工业化，为"reasoning 可验证化"路径提供了完整方法论。

11. **R1-Distill 的"蒸馏胜过 RL"对照**——首次系统性证明对小模型而言，从大模型蒸馏 CoT 比自己跑 RL 更高效，**重塑了 2025 年开源推理小模型的训练范式**。

12. **DeepSeek-OCR 的 Contexts Optical Compression**——把页面图像作为长上下文压缩通道（96%+ 解码精度），与 DSA 在 token 层稀疏化形成互补，是 LLM 上下文扩展的潜在新方向。

13. **混合推理架构（V3.1 起）**——同一 checkpoint 通过 chat template 切换 Think/Non-Think，CoT 长度比 R1-0528 压缩 20–50%；V4 进一步演化为三档思考强度（Non-think / Think High / Think Max），是把推理能力收编进通用基座的关键一步。

---

## 10. 参考文献

### 10.1 通用基座主线（V Series）

1. DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*. arXiv:2401.02954, 2024. https://arxiv.org/abs/2401.02954
2. Dai, D. *et al.* *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. arXiv:2401.06066, 2024. https://arxiv.org/abs/2401.06066
3. DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv:2405.04434, 2024. https://arxiv.org/abs/2405.04434
4. DeepSeek-AI. *DeepSeek-V3 Technical Report*. arXiv:2412.19437, 2024–2025. https://arxiv.org/abs/2412.19437
5. DeepSeek-AI. *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*. arXiv:2512.02556, 2025. https://arxiv.org/abs/2512.02556
6. DeepSeek-AI. *DeepSeek-V4 Technical Report*. HuggingFace, 2026. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
7. DeepSeek-AI. *mHC: Manifold-Constrained Hyper-Connections*. arXiv:2409.19606（更新版 2025-12-31）。
8. *Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention*. arXiv:2506.02523, 2025（第三方 MLA 硬件分析）。

### 10.2 推理模型（R1 Series）

9. DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948 (2025-01-22; v2 2026-01-04). Nature **645**, 633–638 (2025). DOI: 10.1038/s41586-025-09422-z.
10. DeepSeek-AI. *DeepSeek-R1-0528 Release Notes*. https://api-docs.deepseek.com/news/news250528 (2025-05-28).
11. DeepSeek-AI. *DeepSeek-R1-Lite-Preview Announcement* (2024-11). chat.deepseek.com.

### 10.3 视觉语言（VL Series）

12. Lu H., Liu W., Zhang B., et al. *DeepSeek-VL: Towards Real-World Vision-Language Understanding*. arXiv:2403.05525 (2024-03-08).
13. Wu Z., Chen X., Pan Z., et al. *DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding*. arXiv:2412.10302 (2024-12-13).
14. DeepSeek-AI. *DeepSeek-OCR: Contexts Optical Compression*. arXiv:2510.18234 (2025-10-20).

### 10.4 代码（Coder Series）

15. Guo D., Zhu Q., Yang D., et al. *DeepSeek-Coder: When the Large Language Model Meets Programming — The Rise of Code Intelligence*. arXiv:2401.14196 (2024-01-25).
16. Zhu Q., Guo D., Shao Z., et al. *DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence*. arXiv:2406.11931 (2024-06-17).

### 10.5 数学与形式化证明（Math / Prover Series）

17. Shao Z., Wang P., Zhu R., et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300 (2024-02-05).
18. DeepSeek-AI. *DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning*. arXiv:2511.22570 (2025-11-27).
19. Xin H., Guo Z., et al. *DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data*. arXiv:2405.14333 (2024-05-23).
20. Xin H., Ren Z., Song J., et al. *DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search*. arXiv:2408.08152 (2024-08-15).
21. DeepSeek-AI. *DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition*. arXiv:2504.21801 (2025-04-30).

### 10.6 多模态生成（Janus Series）

22. Wu C., Chen X., Wu Z., et al. *Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation*. arXiv:2410.13848 (2024-10-17). CVPR 2025.
23. Ma Y., Liu X., et al. *JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation*. arXiv:2411.07975 (2024-11).
24. Chen X., Wu Z., Liu X., et al. *Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling*. arXiv:2501.17811 (2025-01-27).

### 10.7 官方公告与 Changelog

25. DeepSeek API Docs – Change Log. https://api-docs.deepseek.com/updates （2024-05 ~ 2026-04 全部 API 升级记录）。
26. DeepSeek API Docs – DeepSeek-V3.1 Release (2025-08-21). https://api-docs.deepseek.com/news/news250821
27. DeepSeek API Docs – DeepSeek-V3.1-Terminus (2025-09-22). https://api-docs.deepseek.com/news/news250922
28. DeepSeek API Docs – Introducing DeepSeek-V3.2-Exp (2025-09-29). https://api-docs.deepseek.com/news/news250929
29. DeepSeek API Docs – DeepSeek-V4 Preview Release (2026-04-24). https://api-docs.deepseek.com/news/news260424

### 10.8 第三方综述与交叉验证

30. Sebastian Raschka. *A Technical Tour of the DeepSeek Models from V3 to V3.2*. Ahead of AI Magazine, 2025-12. https://magazine.sebastianraschka.com/p/technical-deepseek
31. Rohan Paul. *DeepSeek-V3's Architectural Revolution*. https://www.rohan-paul.com/p/deepseek-v3-technical-report-they

### 10.9 代码与权重仓库

- GitHub Org: https://github.com/deepseek-ai （34+ repositories）
- Hugging Face Org: https://huggingface.co/deepseek-ai
- 关键 repo：`DeepSeek-LLM`、`DeepSeek-MoE`、`DeepSeek-V2`、`DeepSeek-V3`、`DeepSeek-V3.2-Exp`、`DeepSeek-R1`、`DeepSeek-VL2`、`DeepSeek-Coder-V2`、`DeepSeek-Prover-V2`、`Janus`、`DeepSeek-OCR`。
- HuggingFace V4 集合：https://huggingface.co/collections/deepseek-ai/deepseek-v4

---

> *本报告基于 [deepseek-text-models.md](file:///Users/zhoucanzong/Documents/codes/research-report/papers/DeepSeek-research/deepseek-text-models.md)（纯文本基座模型）与 [deepseek-multimodal-and-specialized.md](file:///Users/zhoucanzong/Documents/codes/research-report/papers/DeepSeek-research/deepseek-multimodal-and-specialized.md)（多模态与专业模型）两份子报告整合提炼。如需查看某一方向的更详细技术细节（如训练超参、消融实验、benchmark 完整数值），请回到子报告。*
