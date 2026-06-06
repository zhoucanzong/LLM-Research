# DeepSeek 多模态与专业模型发展脉络

> 调研范围：DeepSeek 在通用 LLM（V 系列）之外的所有专业方向模型，重点覆盖推理、视觉语言、代码、数学/形式化证明、多模态生成五大主线，以及 2025–2026 年发布的统一/混合模型。
> 调研时点：2026-06。

---

## 全景导览（Executive Summary）

DeepSeek 自 2023 年起以"开源 + 大规模 MoE + 极致训练效率"为旗号，分两条主线推进：

1. **通用基座主线（V-Series）**：DeepSeek-LLM → DeepSeek-V2 → DeepSeek-V3 → V3-0324 → V3.1 → V3.1-Terminus → V3.2-Exp → V3.2 / V3.2-Speciale → V4（Pro/Flash），从 Dense 演进到 MoE，再演进到带稀疏注意力（DSA）和混合注意力（CSA/HCA）的 1M 上下文长程模型。
2. **专业方向主线（本报告核心）**：以 V/MoE 基座为底，针对单一能力极致优化的"垂直旗舰"。

| 方向 | 代表模型 | 关键年份 | 关键贡献 |
| --- | --- | --- | --- |
| 推理 | DeepSeek-R1-Lite / R1-Zero / R1 / R1-0528 / R1-Distill | 2024.11–2025.05 | 纯 RL 激发推理能力（GRPO）、6 个蒸馏小模型、Nature 发表 |
| 视觉语言 | DeepSeek-VL → DeepSeek-VL2 | 2024.03 / 2024.12 | 混合视觉编码器、动态切片 + DeepSeekMoE + MLA |
| 代码 | DeepSeek-Coder → Coder-V2 | 2024.01 / 2024.06 | 仓库级 FIM、338 种语言、对标 GPT-4-Turbo |
| 数学 | DeepSeekMath → DeepSeekMath-V2 | 2024.02 / 2025.11 | 提出 GRPO；自验证数学推理 + IMO 金牌级 |
| 形式化证明 | Prover-V1 → V1.5 → V2 | 2024.05 / 2024.08 / 2025.04 | Lean 4 自动证明、专家迭代、子目标分解 RL |
| 统一多模态生成 | Janus → JanusFlow → Janus-Pro | 2024.10 / 2024.11 / 2025.01 | 解耦视觉编码 = 统一理解 + 生成 |
| 文档/视觉压缩 | DeepSeek-OCR | 2025.10 | 上下文光学压缩，96%+ 精度 |

---

## Part 1 推理模型：DeepSeek-R1 系列（重点）

### 1.1 R1-Lite-Preview（2024 年 11 月）

- 发布形式：在 chat.deepseek.com 上线的预览版（未开源 weights，未发表正式论文）。
- 定位：DeepSeek 第一次正面回应 OpenAI 在 2024 年 9 月发布的 o1-preview。在 AIME、MATH-500 上首次接近 o1-preview 水平。
- 设计意图：验证"长思维链 + 强化学习"路径可行，为 R1 正式版的训练管线探路。

### 1.2 DeepSeek-R1-Zero 与 DeepSeek-R1（2025 年 1 月 22 日）

**论文**：DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948（v1: 2025-01-22；v2: 2026-01-04，Nature 645:633-638, 2025）。

#### R1-Zero — 纯强化学习的"涌现推理"

- **基座**：DeepSeek-V3-Base（671B，37B 激活）。
- **训练范式**：完全跳过监督微调（SFT）冷启动，直接对基座做大规模 RL。这是首个公开证明 *无需任何人类标注推理轨迹* 也能让 LLM 涌现自反思、验证、回溯、动态策略切换等高阶推理行为的开源模型。
- **算法**：使用 DeepSeek 自研的 **GRPO**（Group Relative Policy Optimization，源自 DeepSeekMath，详见 Part 4）。GRPO 的关键改动：
  - 去掉 PPO 的 critic value model，改用 *组内相对优势*：对同一 prompt 采样 G 条响应，用组内归一化得分作 advantage。
  - 显存与算力开销显著低于 PPO，特别适合长 CoT。
- **奖励设计**：完全规则化，避免奖励模型 hacking——
  1. *Accuracy reward*：数学题答案匹配、代码题编译/单测通过；
  2. *Format reward*：必须以 `<think>...</think><answer>...</answer>` 包裹推理过程。
- **观察到的 "Aha Moment"**：模型在训练中段自发出现"等等、让我重新检查……"这类自我修正行为，思考长度随训练步数单调上升。
- **缺陷**：可读性差、中英混杂、有时输出冗长重复。

#### R1 — 多阶段管线，对标 OpenAI o1-1217

为缓解 R1-Zero 的可读性问题，DeepSeek 设计了四阶段训练管线：

1. **冷启动 SFT**：用数千条高质量长 CoT 数据（部分由 R1-Zero 生成后人工修订）对 V3-Base 做轻量 SFT，规范输出格式。
2. **面向推理的 RL**：与 R1-Zero 同样的 GRPO + 规则奖励，并加入语言一致性奖励避免中英混杂。
3. **拒绝采样 + SFT**：用阶段 2 收敛模型生成 ~600K 推理数据 + 200K 通用数据（写作、知识问答、自我认知）一起 SFT，让模型在保留推理能力的同时回归通用对齐。
4. **全场景 RL**：最后再做一轮 RL，目标兼顾 helpfulness/harmlessness（用奖励模型）与 accuracy（用规则奖励），完成最终对齐。
- **训练成本**：根据 Nature 补充材料披露，R1 仅花费约 **US$294K**（不含 V3-Base 约 600 万美元的预训练）。
- **性能**：在 AIME 2024（pass@1 ≈ 79.8%）、MATH-500（97.3%）、Codeforces（2029 ELO，96.3 percentile）、MMLU、GPQA Diamond 等基准上与 OpenAI o1-1217 持平或超越。
- **协议**：MIT License；首个面向 LLM 推理的"完全开源 + 完全可商用"前沿模型。

### 1.3 R1 的蒸馏：把 671B 的"思维"装进小模型

DeepSeek 用 R1 生成 ~80 万条推理样本（≈60 万推理 + 20 万通用），对 6 个开源 dense 基座做单纯 SFT（**故意不做 RL**，旨在测试"知识注入是否足够"）：

| 蒸馏模型 | 基座 | AIME 24 pass@1 | MATH-500 | LiveCodeBench |
| --- | --- | --- | --- | --- |
| R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 28.9 | 83.9 | 16.9 |
| R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 55.5 | 92.8 | 37.6 |
| R1-Distill-Llama-8B | Llama-3.1-8B | 50.4 | 89.1 | 39.6 |
| R1-Distill-Qwen-14B | Qwen2.5-14B | 69.7 | 93.9 | 53.1 |
| R1-Distill-Qwen-32B | Qwen2.5-32B | **72.6** | 94.3 | 57.2 |
| R1-Distill-Llama-70B | Llama-3.3-70B-Instruct | 70.0 | 94.5 | **57.5** |

关键结论：
- **32B 蒸馏版在多项数学/编码基准上超过 OpenAI o1-mini**，是当时社区最强的"消费级硬件可部署"推理模型。
- 论文做了对照实验：直接把 R1 的 RL 管线复用到 Qwen-32B-Base，得到的模型反而显著弱于"用 R1 蒸馏的 Qwen-32B"。说明**对小模型而言，从大模型蒸馏 CoT 比自己跑 RL 更高效**。

### 1.4 DeepSeek-R1-0528（2025 年 5 月 28 日）

- 体量从 671B 微调到 685B（新增 token & 路由专家）。
- 改进点：
  - 思考预算翻倍：AIME 题目平均思考 token 12K → 23K，AIME 2025 准确率 70% → 87.5%。
  - 幻觉降低 45–50%（改写、摘要、阅读理解任务）。
  - 新增 **JSON 输出**与**函数调用**支持；支持 system prompt；不再需要手动塞入 `<think>\n` 触发推理。
  - 前端代码 / vibe-coding 显著增强。
- 同步释出 `DeepSeek-R1-0528-Qwen3-8B` 蒸馏版。

### 1.5 R1 之后：从"双轨"到"混合推理"（2025–2026）

DeepSeek 在 2025 下半年改变了"V3 走通用、R1 走推理"的双模型策略，开始把两者合并为**单一 hybrid 模型**：

- **V3.1（2025-08）**：首个 *Think / Non-Think* 双模式模型，通过切换 chat template 切换 CoT 与直接回答；CoT 长度压缩 20–50%。
- **V3.1-Terminus（2025-09）**：语言一致性、Agent 表现的小修。
- **V3.2-Exp（2025-09）**：引入 **DeepSeek Sparse Attention (DSA)**，长上下文训练/推理效率显著提升。
- **V3.2 / V3.2-Speciale（2025-12）**：V3.2 主打"短链高效"；V3.2-Speciale 是高算力科研版，融合 **DeepSeekMath-V2** 的定理证明能力，2025 年 IMO/IOI 取得金牌级表现。
- **V4-Pro / V4-Flash（2026-04）**：1.6T / 284B MoE，1M 上下文，三档思考强度（Non-think / Think High / Think Max），混合 CSA + HCA 注意力。

> 注：本报告主线是"专业方向"，V 系列详细架构在 DeepSeek 系列调研主报告中展开。

---

## Part 2 视觉语言模型：DeepSeek-VL 系列

### 2.1 DeepSeek-VL（2024 年 3 月，arXiv:2403.05525）

第一代 VL 模型，覆盖 1.3B / 7B 两档 dense LLM 基座。三大设计支柱：

1. **数据 (Data)**：覆盖网页截图、PDF、OCR、图表、知识类内容，并基于真实用户场景构建任务分类法（taxonomy）和指令微调集，强化"实战能力"。
2. **混合视觉编码器 (Hybrid Vision Encoder)**：
   - 高分辨率分支：基于 SAM-B（图像分辨率高达 1024×1024）；
   - 低分辨率分支：基于 SigLIP-L（语义浓缩）；
   - 两路特征经独立 MLP 投影后在通道维拼接，再过一个 MLP 进入 LLM。这一设计兼顾文档级细节（小字、表格）和宏观语义。
3. **三阶段训练 + 视觉-语言均衡**：
   - Stage 1：仅训练 vision-language adaptor；
   - Stage 2：联合预训练，从一开始就把 LLM 一起训，并引入精心设计的视觉/语言 token 比例调度，避免视觉训练侵蚀 LLM 能力；
   - Stage 3：监督微调 + 偏好对齐。
- 1.3B / 7B 在同档 VLM 基准上达到 SOTA 或接近 SOTA，且语言基准退化最小。

### 2.2 DeepSeek-VL2（2024 年 12 月 13 日，arXiv:2412.10302）

围绕"高分辨率 + MoE 高吞吐"做两处重大升级：

#### 视觉侧：动态切片 (Dynamic Tiling)

- 用一个固定基础分辨率的视觉编码器（SigLIP-SO400M-384）处理任意宽高比、任意分辨率图像。
- 输入图被 *切片* 成若干 384×384 patch + 一张 thumbnail，token 序列拼接后送入 LLM。
- 相比 VL 的"双分支固定 1024×1024"，VL2 能处理更长的纵向截图、4K 大图、文档页面，对 OCR/Doc/Chart/Table 任务尤其友好。

#### 语言侧：DeepSeekMoE + MLA

- 改用 DeepSeek V2/V3 同款 **DeepSeekMoE** 架构（共享专家 + 细粒度专家）。
- 引入 **Multi-head Latent Attention (MLA)**：把 KV cache 压缩到低秩 latent 向量，长上下文推理显存降一个数量级。

#### 三个尺寸（按激活参数）

| 模型 | Activated | Total | 备注 |
| --- | --- | --- | --- |
| DeepSeek-VL2-Tiny | 1.0B | 3.4B | 边缘部署 |
| DeepSeek-VL2-Small | 2.8B | 16.1B | 平衡型 |
| DeepSeek-VL2 | 4.5B | 27.5B | 旗舰 |

- **能力**：VQA、OCR、文档/表格/图表理解、视觉定位（grounding）。在同等激活参数下击败大量 dense 与 MoE 开源模型。

### 2.3 DeepSeek-OCR（2025 年 10 月 20 日，arXiv:2510.18234）

- 一个 3B VLM，主线不是"OCR 工具"而是 **Contexts Optical Compression** 的研究：把整页文档压成极少量视觉 token（远少于直接 OCR 后的文本 token），仍保持 96%+ 解码精度。
- 隐含结论：未来可以用"页面图像 → 视觉 token → LLM"的方式 *显著压缩* 长文本上下文，是 LLM 上下文扩展的潜在新方向。

---

## Part 3 代码模型：DeepSeek-Coder 系列

### 3.1 DeepSeek-Coder（2024 年 1 月 25 日，arXiv:2401.14196）

- 体量：1.3B / 5.7B / 6.7B / 33B 共四档 dense 模型，全部 *从零训练* 在 2T token 上（87% 代码 + 13% 自然语言，含中英）。
- 关键技术：
  - **仓库级数据组织**：使用拓扑排序在 *仓库范围* 内拼接多文件，让模型理解跨文件依赖；上下文 16K。
  - **Fill-in-the-Middle (FIM)**：训练时按比例随机执行"前缀-后缀-中段"重排，使模型同时具备 left-to-right 生成与中段补全能力，是 IDE 内联补全的核心技术。
  - **两阶段训练**：通用代码预训练 + 数学/项目级再训练；继之以 instruction tuning 得到 `DeepSeek-Coder-Instruct`。
- 性能：HumanEval、MBPP、DS-1000 上超越 CodeLlama，并击败 Codex / GPT-3.5；33B 版本是当时最强开源代码模型。
- License：宽松商用许可。

### 3.2 DeepSeek-Coder-V2（2024 年 6 月，arXiv:2406.11931）

- 架构：在 **DeepSeek-V2-Base**（MoE）上做"代码继续预训练"，得到两档：
  - DeepSeek-Coder-V2-Lite：16B 总 / 2.4B 激活；
  - DeepSeek-Coder-V2：236B 总 / 21B 激活。
- 关键升级：
  - **支持语言数 86 → 338**；
  - **上下文 16K → 128K**；
  - 继续预训练数据 6T token，FIM 策略保留并优化；
  - 后训练阶段加入 *DPO* + *GRPO*（数学/代码 RL，与 DeepSeekMath 同源）。
- 性能：HumanEval 90.2、MBPP+ 76.2、LiveCodeBench、SWE-Bench 等基准上**首次让开源 MoE 在代码智能上对标 GPT-4-Turbo / Claude-3-Opus / Gemini-1.5-Pro**。
- 后续：在 2024 年 9 月，DeepSeek 发布 **V2.5**，把 V2-Chat 与 Coder-V2 合并升级，标志着"通用 + 代码"分支再度合流，预示 V3 起代码能力将通过通用基座 + 后训练的方式提供，独立 Coder 系列暂未见 V3 形态。
- V3 / V3.1 / V3.2 在 LiveCodeBench、SWE-Bench Verified、Codeforces 等代码与 Agent benchmark 上持续推进，事实上成为 DeepSeek 当前的"代码主战场"。

---

## Part 4 数学与形式化证明

### 4.1 DeepSeekMath（2024 年 2 月 5 日，arXiv:2402.03300）

- 7B dense 模型，从 DeepSeek-Coder-Base 7B 出发继续预训练于 120B 数学 token（高质量爬取 Common Crawl + 自训练分类器去噪）。
- 在 MATH 上以 7B 体量实现 51.7%，逼近 GPT-4 (52%) 与 Gemini-Ultra。
- **最大贡献：提出 GRPO**（Group Relative Policy Optimization）：
  - 去掉 critic，对同一提示采样 G 条 rollout，组内归一化作 advantage；
  - 节省接近一半显存与计算；
  - 后续被 R1、Coder-V2、Prover-V1.5/V2、社区开源训练框架（OpenRLHF、TRL、verl 等）广泛采用。
- 这篇论文事实上是后来 **R1 训练算法的"前传"**。

### 4.2 DeepSeekMath-V2（2025 年 11 月 27 日，arXiv:2511.22570）

- **目标转向"自验证"（self-verifiable）数学推理**：模型不仅要给答案，还要生成可由模型自己复核的*证明*；训练目标也从"答案匹配"转为"证明合格"，从根本上抑制"对答案胡编"的奖励作弊。
- 体现两个新方向：
  1. *Process-level verification*：把奖励信号从最终答案搬到推理过程中的每个步骤；
  2. *Theorem-proving 与一般数学推理融合*：吸收 Prover 系列的形式化证明能力。
- 在 IMO 2025、CMO 2024 上取得**金牌级**得分；其能力被并入 V3.2-Speciale。

### 4.3 DeepSeek-Prover 系列（Lean 4 形式化定理证明）

| 版本 | 时间 | arXiv | 关键贡献 |
| --- | --- | --- | --- |
| **Prover-V1** | 2024-05-23 | 2405.14333 | 首版：从高中/竞赛题自动 *autoformalize* 为 Lean 4 statement，再用模型生成证明，用 Lean 4 验证作为 ground truth；筛选出 8M 高质量合成证明对 7B 模型做 SFT，miniF2F-test 达 50.0% |
| **Prover-V1.5** | 2024-08-15 | 2408.08152 | 继续预训练 + RLPAF（用 Lean proof-assistant 反馈做 RL）+ **RMaxTS**（Monte-Carlo Tree Search 引入 intrinsic reward 鼓励多样路径）。miniF2F 63.5%，ProofNet 25.3% |
| **Prover-V2** | 2025-04-30 | 2504.21801 | 671B（基于 V3）+ **子目标分解**：先用大模型把非形式化证明拆成 subgoal，再让 7B prover 逐个完成、最后回拼。引入 *cold-start RL*：从一批 high-quality 证明开始，用 GRPO 微调；miniF2F 88.9%，攻克 PutnamBench 的相当一部分题；同时发布 7B 与 671B 双尺寸 |

形式化证明系列与 DeepSeekMath-V2 的"自验证"思想高度一致，可视为 DeepSeek 在"reasoning 可验证化"路径上的两个互补支柱。

---

## Part 5 多模态生成：Janus 系列

### 5.1 Janus（2024 年 10 月 17 日，arXiv:2410.13848）

> 1.3B 参数（DeepSeek-LLM-1.3B 基座），首次给"统一理解 + 生成"模型一个清晰的架构指引。

**核心思想：解耦视觉编码 (Decoupling Visual Encoding)**

之前的"Chameleon 路线"用 *单一* 视觉 tokenizer 同时承担"理解"（需高语义抽象）与"生成"（需细粒度像素级 token），两者目标冲突，导致理解性能下滑。Janus 的解法：

| 任务 | 视觉编码器 | 输出空间 |
| --- | --- | --- |
| 多模态理解 | SigLIP-L（连续语义特征） | 投影到 LLM token 空间 |
| 图像生成 | VQ tokenizer（离散像素 token） | 自回归生成离散 image token |

- 两路 *分别* 编码，但**共享同一个 Transformer 主干**做下游建模。架构因此既统一又解耦。
- 训练分三阶段：adaptor 训练 → 统一预训练 → 指令微调。
- 在 GenEval、DPG-Bench 等图像生成基准与 MMBench 等理解基准上，**同时**优于此前的 unified models（Chameleon、Show-o）。

### 5.2 JanusFlow（2024 年 11 月）

- 把生成路径换成 **Rectified Flow**（连续流匹配），抛弃 VQ tokenizer，进一步提升生成质量与采样效率，仍延续解耦视觉编码思路。

### 5.3 Janus-Pro（2025 年 1 月 27 日，arXiv:2501.17811）

- 1B 与 7B 双尺寸，相对 Janus 三处升级：
  1. **优化训练策略**：调整三阶段时长、视觉 token 比例；
  2. **扩展数据**：理解侧多模态 SFT 数据扩到 ~9000 万样本；生成侧加入 7200 万合成审美数据；
  3. **模型放大**至 7B。
- 在 GenEval、DPG-Bench 上**一度超过 SDXL、DALL·E 3**，在春节期间引爆了 Hugging Face 趋势榜，是 2025 年初的现象级开源生成模型。
- 与 R1 在同一周发布，构成 DeepSeek 2025 年初的"双爆点"。

---

## Part 6 DeepSeek 模型生态全景与时间线

### 6.1 时间线（2023.11 – 2026.04）

```
2023-11  DeepSeek-LLM-7B/67B（首发，Dense）
2024-01  DeepSeek-Coder（1.3B–33B Dense）
2024-01  DeepSeek-MoE-16B（首款 MoE）
2024-02  DeepSeekMath（7B，提出 GRPO）
2024-03  DeepSeek-VL（1.3B/7B）
2024-05  DeepSeek-V2（236B/21B MoE，MLA）
2024-05  DeepSeek-Prover-V1
2024-06  DeepSeek-Coder-V2（236B/21B MoE）
2024-08  DeepSeek-Prover-V1.5
2024-09  DeepSeek-V2.5（V2-Chat ⊕ Coder-V2 合并）
2024-10  Janus（1.3B 统一理解+生成）
2024-11  JanusFlow；R1-Lite-Preview（线上）
2024-12  DeepSeek-V3（671B/37B）；DeepSeek-VL2（Tiny/Small/Base）
─────────────────────────  2025  ─────────────────────────
2025-01  DeepSeek-R1-Zero、R1（arXiv 2501.12948）
2025-01  6 个 R1-Distill（Qwen 1.5/7/14/32B + Llama 8/70B）
2025-01  Janus-Pro（1B/7B）
2025-03  DeepSeek-V3-0324
2025-04  DeepSeek-Prover-V2（7B + 671B）
2025-05  DeepSeek-R1-0528 + R1-0528-Qwen3-8B
2025-08  DeepSeek-V3.1（Hybrid Think / Non-Think）
2025-09  V3.1-Terminus；V3.2-Exp（DSA）
2025-10  DeepSeek-OCR（光学上下文压缩）
2025-11  DeepSeekMath-V2（自验证证明）
2025-12  DeepSeek-V3.2 / V3.2-Speciale
─────────────────────────  2026  ─────────────────────────
2026-01  R1 论文 v2（Nature 645:633-638）
2026-04  DeepSeek-V4-Pro / V4-Flash（1M context, CSA+HCA）
```

### 6.2 模型族关系图

```
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


  视觉/多模态:    DeepSeek-LLM-1.3B / 7B  ──► VL (hybrid encoder)
                                          └► Janus / JanusFlow / Janus-Pro (decoupled enc.)

                      DeepSeek-V2-MoE     ──► VL2 (dynamic tiling + MLA)
                      DeepSeek-V3         ──► OCR (contexts optical compression)


  代码:         Coder (Dense) ──► Coder-V2 (MoE on V2-Base) ──► 并入 V2.5/V3 主线


  数学/证明:    DeepSeekMath ──► GRPO ──► R1 / Coder-V2-RL
                       └─► DeepSeekMath-V2 (self-verifiable)
                Prover-V1 ──► V1.5 (RLPAF + RMaxTS) ──► V2 (subgoal decomp + cold-start RL)
```

### 6.3 五个贯穿全系列的"DeepSeek 方法学"关键词

1. **MoE + 共享专家 + 细粒度专家 (DeepSeekMoE)**：从 V2 起成为所有大模型的默认架构，被 VL2、Coder-V2、R1、V3.x 复用。
2. **MLA (Multi-head Latent Attention)**：长上下文推理的关键，是 V2/VL2/V3 大幅降低 KV cache 的核心机制。
3. **GRPO**：从 DeepSeekMath 提出，到 R1 验证为"无 SFT 也能涌现推理"的训练算法，已经成为开源 RL 训练事实上的工业标准之一。
4. **数据分类法 + 真实场景驱动**：VL、Coder、Math、R1 都强调"从真实使用场景反推数据 taxonomy"，而非堆基准。
5. **自验证 / 形式化奖励**：R1 用规则奖励、Prover 用 Lean、Math-V2 用 process verifier，DeepSeek 一以贯之地避免"用奖励模型 hack 奖励模型"。

---

## 论文与发布引用列表

### 推理
- DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948 (2025-01-22; v2 2026-01-04). Nature **645**, 633–638 (2025). DOI: 10.1038/s41586-025-09422-z.
- DeepSeek-AI. *DeepSeek-R1-0528 Release Notes*. https://api-docs.deepseek.com/news/news250528 (2025-05-28).
- DeepSeek-AI. *DeepSeek-R1-Lite-Preview Announcement* (2024-11). chat.deepseek.com.

### 视觉语言
- Lu H., Liu W., Zhang B., et al. *DeepSeek-VL: Towards Real-World Vision-Language Understanding*. arXiv:2403.05525 (2024-03-08).
- Wu Z., Chen X., Pan Z., et al. *DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding*. arXiv:2412.10302 (2024-12-13).
- DeepSeek-AI. *DeepSeek-OCR: Contexts Optical Compression*. arXiv:2510.18234 (2025-10-20).

### 代码
- Guo D., Zhu Q., Yang D., et al. *DeepSeek-Coder: When the Large Language Model Meets Programming — The Rise of Code Intelligence*. arXiv:2401.14196 (2024-01-25).
- Zhu Q., Guo D., Shao Z., et al. *DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence*. arXiv:2406.11931 (2024-06-17).

### 数学与形式化证明
- Shao Z., Wang P., Zhu R., et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300 (2024-02-05).
- DeepSeek-AI. *DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning*. arXiv:2511.22570 (2025-11-27).
- Xin H., Guo Z., et al. *DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data*. arXiv:2405.14333 (2024-05-23).
- Xin H., Ren Z., Song J., et al. *DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search*. arXiv:2408.08152 (2024-08-15).
- DeepSeek-AI. *DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition*. arXiv:2504.21801 (2025-04-30).

### 多模态生成（Janus）
- Wu C., Chen X., Wu Z., et al. *Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation*. arXiv:2410.13848 (2024-10-17). CVPR 2025.
- Ma Y., Liu X., et al. *JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation*. arXiv:2411.07975 (2024-11).
- Chen X., Wu Z., Liu X., et al. *Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling*. arXiv:2501.17811 (2025-01-27).

### 通用基座主线（用于交叉参考）
- DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv:2405.04434 (2024-05).
- DeepSeek-AI. *DeepSeek-V3 Technical Report*. arXiv:2412.19437 (2024-12).
- DeepSeek-AI. *DeepSeek-V3.1 / V3.2-Exp / V3.2 / V3.2-Speciale Release Notes*. https://api-docs.deepseek.com/updates (2025-08 → 2025-12).
- DeepSeek-AI. *DeepSeek-V4 Release Notes*. (2026-04).

### 代码与权重仓库
- GitHub Org: https://github.com/deepseek-ai (34+ repositories)
- Hugging Face Org: https://huggingface.co/deepseek-ai
- 关键 repo：`DeepSeek-R1`, `DeepSeek-VL2`, `DeepSeek-Coder-V2`, `DeepSeek-Prover-V2`, `Janus`, `DeepSeek-OCR`。

---

> *本报告聚焦 DeepSeek 的多模态与专业方向模型，与"DeepSeek 通用 LLM 主线（V 系列）"调研报告互为补充。两份文档共同构成 DeepSeek 系列完整调研的基础素材。*
