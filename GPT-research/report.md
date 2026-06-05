# OpenAI GPT 系列模型深度调研报告

> 调研对象：OpenAI 自 2018 年至 2026 年发布的 Generative Pre-trained Transformer（GPT）系列与 o-Reasoning 系列模型
> 时间跨度：2018 年 6 月 ~ 2026 年 5 月
> 资料来源：OpenAI 论文（arXiv）、OpenAI 官方研究页（openai.com/research）、官方博客（openai.com/blog）、System Card、API 文档、第三方逆向分析与学界综述
> 撰写日期：2026 年 6 月
> 重要说明：OpenAI 自 GPT-4 起几乎完全闭源，**模型规模、架构、训练数据组成、对齐细节大量未公开**。本报告对 **可确认（Confirmed）**、**官方暗示（Implied）**、**社区推测（Speculated）** 三类信息做明确标注；任何带 *推测* 标签的内容均不作为权威结论。

---

## 目录

- [1. 概述与时间线](#1-概述与时间线)
- [2. 基座模型演进：GPT-1 → 2 → 3 → 4 → 4o → 4.1 → 5](#2-基座模型演进gpt-1--2--3--4--4o--41--5)
- [3. 对齐技术演进：从 SFT 到 RLHF 到推理对齐](#3-对齐技术演进从-sft-到-rlhf-到推理对齐)
- [4. 推理模型演进：o1 → o3 → o4-mini](#4-推理模型演进o1--o3--o4-mini)
- [5. 多模态演进：GPT-4V → GPT-4o → Realtime → Sora](#5-多模态演进gpt-4v--gpt-4o--realtime--sora)
- [6. 跨代技术演进分析](#6-跨代技术演进分析)
- [7. 关键创新点（按行业影响力排序）](#7-关键创新点按行业影响力排序)
- [8. 参考文献](#8-参考文献)

---

## 1. 概述与时间线

OpenAI 的 GPT 家族是过去八年大模型行业的主轴线。从 2018 年 117M 参数的 GPT-1 起步，到 2025 年的 GPT-5 与 o4 级推理模型，OpenAI 完成了至少四次范式级跃迁：

1. **预训练 + 微调（GPT-1, 2018）**：把 NLP 从「针对每个任务训练特定模型」拉向「共用一个 LM 底座」。
2. **预训练 + 提示（GPT-2/3, 2019–2020）**：以 zero-shot / few-shot prompting 取代显式微调，确立 in-context learning（ICL）。
3. **预训练 + 对齐（InstructGPT/ChatGPT, 2022）**：SFT + RLHF 把 raw LM 转化为可商用的对话助手，引爆 ChatGPT 现象。
4. **预训练 + 推理（o1/o3/o4, 2024–2025）**：以 RL on chain-of-thought 把 *test-time compute* 作为新的 scaling 维度。

并行还有两条次要主线：
- **多模态主线**：GPT-4V（图像理解）→ GPT-4o（原生 omnimodal 端到端音频/视觉）→ Sora（视频生成）→ GPT-5（统一推理 + 多模态）。
- **效率/小模型主线**：GPT-3.5 Turbo → GPT-4o mini → GPT-4.1 mini/nano → o4-mini，把同代旗舰能力压缩到更低 API 成本与端侧场景。

### 1.1 关键时间线（精选）

| 时间 | 模型 / 事件 | 类型 | 关键定位（已公开信息为主） |
|---|---|---|---|
| 2018-06 | **GPT-1** | LLM | 12 层 Transformer Decoder，117M 参，BookCorpus 预训练 + 任务微调 |
| 2019-02 | **GPT-2**（先 124M，后 1.5B） | LLM | WebText 40GB 训练，宣称「零样本多任务学习者」，分阶段开放 |
| 2020-05 | **GPT-3** (davinci-175B)，arXiv:2005.14165 | LLM | 175B 参数，**确立 in-context few-shot learning** |
| 2021-01 | DALL·E 1 / CLIP | 多模态 | 文本-图像对比学习，与 GPT 主线分支 |
| 2021-07 | **Codex**（code-davinci-002 前身） | 代码 | 在 GPT-3 上继续训练 GitHub 代码，驱动 GitHub Copilot |
| 2022-01 | **InstructGPT**，arXiv:2203.02155 | 对齐 | SFT + RLHF（PPO）三步法，确立「对齐」工程化范式 |
| 2022-11-30 | **ChatGPT**（gpt-3.5-turbo 前身） | 产品 | 基于 GPT-3.5 + RLHF + 对话 SFT，5 天破百万用户 |
| 2023-03-14 | **GPT-4** Technical Report | LLM | 多模态（图+文输入），**架构与规模官方未披露** |
| 2023-09 | **GPT-4V(ision)** System Card | 多模态 | 把 GPT-4 的视觉输入正式开放 API |
| 2023-11-06 | **GPT-4 Turbo**（gpt-4-1106-preview） | LLM | 128K 上下文、JSON Mode、知识截止 2023-04 |
| 2024-05-13 | **GPT-4o**（"omni"）System Card | 多模态 | 单模型端到端处理 text/audio/image/video，语音延迟 ~320ms |
| 2024-07-18 | **GPT-4o mini** | 小模型 | 替代 GPT-3.5 Turbo，**MMLU 82%**，价格 $0.15/$0.60 per 1M |
| 2024-09-12 | **o1-preview / o1-mini**（"Strawberry"） | 推理 | 首次产品化 *Long CoT + RL*，竞赛数学/科学推理大幅提升 |
| 2024-12-05 | **o1 (full) / o1-pro** | 推理 | o1 正式版上线 ChatGPT Pro $200/月套餐，支持 ProSearch |
| 2024-12-20 | **o3 / o3-mini** 预览（外部红队） | 推理 | ARC-AGI 公开测试集 87.5%（high-compute），震动学界 |
| 2025-01-31 | **o3-mini** 正式上线 API | 推理 | low/medium/high 三档 reasoning effort |
| 2025-04 | **o3 / o4-mini** 全量开放 | 推理 | 首次原生 tool-use（python、search、image）的推理模型 |
| 2025-04-14 | **GPT-4.1 / 4.1 mini / 4.1 nano** | LLM | 1M tokens 上下文；coding / IF / long-context 三轴提升 |
| 2025-08-07 | **GPT-5**（GPT-5 / GPT-5 mini / GPT-5 nano / GPT-5 thinking） | 旗舰 | 「统一模型」首次将 chat 模型与 reasoning 模型合并到单 router 下 |
| 2025-Q4 | GPT-5 系列后续（Codex GPT-5、GPT-5.1） | 旗舰 | 强化 agentic / coding 能力（仅在官方博客披露） |
| 2026-Q1（披露区间） | o4 / GPT-5.5 系列后续 | 推理/旗舰 | 公开信息有限，*推测* 主要是 RL 数据规模与 tool-use 上的迭代 |

> 注：2025 年下半年起 OpenAI 进入「**多模型并行 + 路由**」节奏，official changelog 不再把每个新版本作为「新一代模型」单独命名，而是通过日期戳（如 `gpt-5-2025-11-xx`）滚动迭代。

### 1.2 研究方法与可验证性

OpenAI 的特殊性在于：
- **GPT-1/2/3** 有完整 arXiv 论文与（部分）开源权重，是确切可复现的；
- **InstructGPT** 有完整论文（arXiv:2203.02155），但 RLHF 用的奖励模型与策略模型权重未公开；
- **GPT-4 / 4o / o1 / GPT-5** 仅有 *Technical Report* 或 *System Card*，**刻意省略架构、参数量、数据集组成、训练算力**，只保留评测与安全章节；
- 因此 GPT-4 之后所有「参数量」「MoE 专家数」「训练 token 数」断言都属于第三方推测（如 SemiAnalysis、Geohot、George Hotz、Dylan Patel 等），本报告会标注 *推测*。

---

## 2. 基座模型演进：GPT-1 → 2 → 3 → 4 → 4o → 4.1 → 5

### 2.1 GPT-1（2018）：预训练 + 微调范式的奠基

**论文**：Alec Radford et al., *Improving Language Understanding by Generative Pre-Training*, OpenAI Tech Report, 2018-06-11。

GPT-1 并非模型规模意义上的突破，而是**方法论意义上的范式确立**。在它之前，NLP 主流是 ELMo（双向 LSTM 语境向量）、Word2Vec 等表征学习；BERT 尚未发布。GPT-1 首次系统验证了：

- **架构**：12 层 Transformer Decoder（仅 masked self-attention），768 hidden / 12 heads / 3072 FFN，约 **117M 参数**（确认）。
- **预训练目标**：标准 left-to-right autoregressive language modeling，loss 为下一个 token 的负对数似然。
- **预训练数据**：BookCorpus（约 7000 本未出版小说，~5GB，~800M words）。
- **下游适配**：在 GLUE 风格的 12 个 NLP 任务（NLI、QA、分类、相似度）上做有监督微调，**任务输入做轻度结构化**（用特殊 token 拼接 premise/hypothesis），保持模型架构基本不变。

**关键创新点**：
1. 把 Transformer Decoder（来自 Vaswani 2017）首次用于「**单纯生成式预训练**」，而非翻译。
2. 提出 *task-agnostic pre-training + task-specific fine-tuning* 的两阶段范式，为后续整个 LLM 时代奠基。
3. 在 9/12 任务上达到 SOTA，证明「通用语言模型先验」对下游任务的迁移价值。

**局限**：
- 单向（causal）注意力被同年 BERT 的双向 MLM 暂时压制；2018 年底 ~ 2019 年初的 NLP 主流仍是 BERT 派系。
- 微调阶段需要每个任务一份模型副本，工程不可扩展。

### 2.2 GPT-2（2019）：零样本多任务的首次出现

**论文**：Alec Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI Tech Report, 2019-02-14。

GPT-2 的核心贡献是**首次系统提出「足够大的语言模型本身就是多任务学习者」**。它的具体配置（确认）：

- **架构**：依然是 Transformer Decoder，将 LayerNorm 移到子模块输入端（pre-LN）、最后一层加额外 LN，扩展词表到 50,257 BPE token、上下文长度 1024。
- **规模**：从 117M（small）到 **1.5B**（XL），共 4 档；XL 比 GPT-1 大 ~13×。
- **数据**：**WebText**，由 Reddit 高赞外链爬取并清洗的 ~40GB 文本（~8M 文档），主动避开 Wikipedia 以减少与下游评测污染。
- **训练**：BPE on bytes（避免 OOV）、batch size 512、LM 目标无变化。

**关键观察**：
- **零样本（zero-shot）评估**：不微调、不提供示例，直接把任务以自然语言提示（`TL;DR:`、`Translate to French:` 等）描述给模型，在 LAMBADA、CBT、Winograd、Children's Book、阅读理解等多任务上达到或接近当时 SOTA。
- **Scaling 趋势的首次外显**：论文图 4 用模型大小 vs 各任务零样本准确率画出**单调上升曲线**，是后来 *scaling law* 的口语化前奏。

**「分阶段开放」事件**：
2019-02 OpenAI 仅放出 124M 与 355M，称担心被滥用造谣；后于 2019-08 放 774M、2019-11 放 1.5B 全部权重。这一争议事件首次把 LLM 安全/治理推上行业议题，也奠定了后续 GPT-3 不开源、GPT-4 完全闭源的政策走向。

### 2.3 GPT-3（2020）：175B 与 In-Context Learning

**论文**：Tom B. Brown et al., *Language Models are Few-Shot Learners*, **arXiv:2005.14165**, 2020-05-28（NeurIPS 2020 Best Paper）。

GPT-3 是第一篇**只字未提微调**的旗舰 LLM 论文：所有评测全部用 in-context few-shot，凭模型规模一举把 NLP benchmark 推到 SOTA 水位。其规格（确认）：

| 名称 | 层数 | hidden | heads | 参数量 | 学习率 |
|---|---|---|---|---|---|
| GPT-3 Small | 12 | 768 | 12 | 125M | 6e-4 |
| GPT-3 Medium | 24 | 1024 | 16 | 350M | 3e-4 |
| GPT-3 Large | 24 | 1536 | 16 | 760M | 2.5e-4 |
| GPT-3 XL | 24 | 2048 | 24 | 1.3B | 2e-4 |
| GPT-3 2.7B | 32 | 2560 | 32 | 2.7B | 1.6e-4 |
| GPT-3 6.7B | 32 | 4096 | 32 | 6.7B | 1.2e-4 |
| GPT-3 13B | 40 | 5140 | 40 | 13B | 1e-4 |
| **GPT-3 175B (davinci)** | **96** | **12288** | **96** | **175B** | **0.6e-4** |

**训练数据**（确认）：CommonCrawl（filtered，410B tokens，权重 60%）+ WebText2（19B，22%）+ Books1（12B，8%）+ Books2（55B，8%）+ Wikipedia（3B，3%），共 **~300B token** 训练量（按权重不是按 epoch）。注意 175B 模型只看了 ~300B token，远未达 Chinchilla optimal（后述）。

**关键发现**：
1. **In-context learning（ICL）作为新能力**：模型可以通过 prompt 中的若干 demonstration 学到 task pattern，无需更新权重。这是 *涌现能力* 的最早系统案例之一。
2. **缩放预测**：训练损失随模型尺寸 / 数据 / compute 呈幂律下降，与 Kaplan 等 *Scaling Laws for Neural Language Models* (arXiv:2001.08361) 一致。
3. **Few-shot >> Zero-shot >> Fine-tuned baselines**（在多数任务上）：175B 的 1-shot/few-shot 在 SuperGLUE、TriviaQA、LAMBADA 等接近或超过 BERT-Large 微调结果。

**局限**（OpenAI 自己也承认）：
- **算术、常识 / 物理推理** 远逊于人类；
- **bias / hallucination** 大量出现；
- **训练-测试污染**（部分基准被预训练数据污染）OpenAI 单独写章节披露。

### 2.4 GPT-3.5 / Codex 时期（2021–2022）：在产品上换代

GPT-3.5 系列（包括 `text-davinci-002/003`、`code-davinci-002`、`gpt-3.5-turbo`）**没有专门论文**，OpenAI 仅在 API 文档与 blog 中描述差异：

- **Codex**（2021-07，arXiv:2107.03374）：基于 GPT-3 12B 在 GitHub 159GB Python 代码上继续预训练；驱动 GitHub Copilot；评测 HumanEval。这是 **代码模型与基座模型分叉** 的首次正式落地。
- **text-davinci-002**（2022-01）：**FeedME**（人工示范的 SFT，*Implied*）。
- **text-davinci-003**（2022-11）：在 002 基础上加入 PPO RLHF（*Implied*）。
- **gpt-3.5-turbo**（2023-03）：参数大幅缩小（*推测* 约为 175B 的 ~1/10 量级，社区估计 *未确认*），但通过更多 RLHF + 对话 SFT 在对话场景上反而胜过 davinci。

> 这一时期的真正历史意义在于「ChatGPT 作为产品」（详见 §3），而非基座的算法跃迁。

### 2.5 GPT-4（2023-03）：闭源旗舰与多模态首次入场

**资料**：OpenAI, *GPT-4 Technical Report*, arXiv:2303.08774, 2023-03-14；同步 *GPT-4 System Card*。

GPT-4 是 OpenAI **第一份「明确不公开技术细节」** 的旗舰：

> "Given both the competitive landscape and the safety implications of large-scale models like GPT-4, this report contains no further details about the architecture (including model size), hardware, training compute, dataset construction, training method, or similar."

可确认的事实：
- **多模态输入**：支持 image + text 输入，输出仍是 text；视觉能力 2023-09 通过 GPT-4V 系统卡正式开放。
- **上下文**：标准版 8K，gpt-4-32k 提供 32K（之后 128K Turbo 取代）。
- **训练数据截止**：2021-09。
- **能力**：在律师资格考试（UBE）、AP 考试、GRE 等专业考试上跨入人类前 10%；HumanEval 67%（无 SC、单次采样）。
- **预测式 scaling**：论文披露 OpenAI **用 1000× 较小算力的模型外推预测** GPT-4 的 final loss，误差极小，是 *scaling law 工程化* 的首次披露。

广为流传但**未经官方确认**的描述（仅供参考）：
- *推测* 总参数 ~1.8T，**8 路 MoE × 220B 专家**（Geohot/SemiAnalysis 2023-07 披露）；
- *推测* 训练 token ~13T，使用 ~25,000 张 A100 训练 90~100 天；
- *推测* 训练成本 ~$63M。

**这些数字未被 OpenAI 任何官方文件背书。** 行业引用时一般标注为「广泛流传的估计」。

### 2.6 GPT-4 Turbo（2023-11）：长上下文与 JSON 模式

2023 年 11 月 OpenAI DevDay 发布 `gpt-4-1106-preview`：
- **128K 上下文**（首次跨入十万 token 级，与同期 Claude 2.1 200K 错位）；
- **JSON Mode** + **Function Calling 升级**：原生结构化输出，工程上替代 LangChain 的多数包装；
- **知识截止前推到 2023-04**；
- **成本下降 ~3×**（输入 $0.01/1K，输出 $0.03/1K）。

技术上 *推测* 是同代基座的蒸馏 + 长上下文继续训练，未发表正式论文。

### 2.7 GPT-4o（2024-05）：原生 Omnimodal

**资料**：OpenAI, *Hello GPT-4o*, 2024-05-13；*GPT-4o System Card*, 2024-08-08。

GPT-4o 的「o」代表 **omni**，是 OpenAI 第一个**单模型端到端处理 text/audio/image/video** 的旗舰：

- **原生音频**：以前的 ChatGPT Voice 是 ASR (Whisper) → GPT-4 → TTS 三段串联，端到端延迟 2.8~5.4 s；GPT-4o 直接以**音频 token** 作为输入输出，平均延迟 ~**320 ms**，逼近人类对话节奏（确认）。
- **多模态融合方式**（*Implied + 推测*）：模型同时接收 *unified token stream*（文字、视觉 patch、音频帧均映射到同一序列），共享 Transformer 计算；System Card 提到「single model trained end-to-end across text, vision, and audio」。
- **能力指标**（确认）：MMLU 88.7、HumanEval 90.2、MGSM（多语言数学）90.5，在英语外语种上明显优于 GPT-4 Turbo。
- **价格**：API 单价较 GPT-4 Turbo 砍半至输入 $5/1M，输出 $15/1M。
- **GPT-4o mini**（2024-07）：MMLU 82%，价格 $0.15/$0.60，正式取代 gpt-3.5-turbo。

**未公开**：参数量、激活策略、token 化方式（视觉/音频 codebook 的设计）、训练数据。社区基于 inference 速度与价格 *推测* 总参数低于 GPT-4，但激活更高效（可能是 MoE 的某种瘦身或 dense 重构），**无任何官方信息**。

### 2.8 GPT-4.1（2025-04）：上下文与编码的工程季

**资料**：OpenAI, *Introducing GPT-4.1 in the API*, 2025-04-14。

GPT-4.1 不再是「下一代基座」而是「同代工程冲刺」：

- **1M tokens 上下文**（输入），输出仍为 32K；
- **Coding**：SWE-bench Verified 54.6%（从 GPT-4o 33.2% 提升），Aider Polyglot 51.6%；
- **Instruction Following**：MultiChallenge 38.3%（GPT-4o 27.8%）；
- **Long-context**：MRCR 上 1M token 检索能保持高准确率（OpenAI 自评）；
- **三档定价**：GPT-4.1 / 4.1 mini / 4.1 nano，价格分别为 $2.0/$8.0、$0.4/$1.6、$0.1/$0.4 per 1M token；nano 是 OpenAI 首次推出的「真·端侧候选」尺寸。

**未公开**：参数量、训练数据组成；OpenAI 官方仅称「improved post-training and tool-use distillation」。

### 2.9 GPT-5（2025-08）：统一推理 + 路由

**资料**：OpenAI, *Introducing GPT-5*, 2025-08-07；*GPT-5 System Card*, 2025-08-07。

GPT-5 的最大变化不在于「模型变大」，而在于**模型形态的变化**：

- **统一模型**：GPT-5 不是单一权重，而是 `gpt-5-main / gpt-5-thinking / gpt-5-mini / gpt-5-nano` 等多权重在一个 **router** 下协同。Router 根据 query 难度自动决定是否切到 *thinking*（即原 o 系推理模型）。这相当于把 chat 模型和推理模型合并为一个产品形态。
- **能力**（确认）：
  - HumanEval 替代品 SWE-bench Verified **74.9%**（GPT-4.1 54.6%）；
  - AIME 2025 **94.6%**（无工具）/ **100%**（with tools，Pass@1）；
  - Multimodal MMMU 84.2%；
  - HealthBench Hard 46.2%。
- **幻觉率**：在事实问答上比 GPT-4o 下降 ~45%（GPT-5）/ ~80%（GPT-5 thinking），official blog 公布。
- **可控性**：API 暴露 `verbosity`（low/medium/high）与 `reasoning_effort`（minimal/low/medium/high）两组旋钮。

**未公开 / 推测**：参数量、专家数、上下文长度（API 上目前 400K input + 128K output，*已确认*但比 GPT-4.1 的 1M 反而缩水）、router 实现细节（*推测* 是一个轻量分类器，可能含一次低 cost forward）。

GPT-5 之后 OpenAI 进入滚动版本节奏（gpt-5-2025-09-xx、Codex GPT-5、GPT-5.1 等），不再做发布会式跃迁，更多是「模型路由 + 评测线」工程迭代。

### 2.10 Scaling 路径的总览

把上述参数 / 数据 / 算力放在同一张表，能看到 OpenAI 公开 vs 不公开的清晰断裂：

| 模型 | 年份 | 参数（M=百万，B=十亿） | 训练 token | 训练算力 (FLOPs) | 数据来源 | 公开级别 |
|---|---|---|---|---|---|---|
| GPT-1 | 2018 | 117 M | ~1.3B | ~2.6e19 | BookCorpus | 论文 + 权重开源 |
| GPT-2 (1.5B) | 2019 | 1.5 B | ~10B | ~1.5e21 | WebText 40GB | 论文 + 权重开源 |
| GPT-3 (175B) | 2020 | 175 B | 300 B | ~3.14e23 | CC + Web + Books | 论文，无权重 |
| GPT-3.5 / Codex | 2021–22 | *推测* 175B / 12B | *推测* > 300B | 未公开 | + GitHub | API only |
| GPT-4 | 2023 | *推测* ~1.8T MoE | *推测* ~13T | *推测* ~2e25 | 未公开 | Tech Report，无细节 |
| GPT-4 Turbo | 2023.11 | 同 GPT-4 | 增量 | 未公开 | + 2023 数据 | API only |
| GPT-4o | 2024.05 | 未公开 | 未公开 | 未公开 | + 音频 / 视觉 | System Card |
| GPT-4o mini | 2024.07 | 未公开（小） | 未公开 | 未公开 | 未公开 | 仅 API |
| GPT-4.1 | 2025.04 | 未公开 | 未公开 | 未公开 | 未公开 | 博客 |
| GPT-5 | 2025.08 | 未公开（统一路由） | 未公开 | 未公开 | 未公开 | System Card |

> **观察 1**：从 GPT-1 到 GPT-3，参数量 1500×、算力 ~10⁴×，loss 几乎严格遵循 Kaplan scaling law。
> **观察 2**：GPT-3 → GPT-4 *预测式* scaling 是 OpenAI 公开的最重要工程方法；之后 OpenAI 把 scaling law 视作核心商业秘密。
> **观察 3**：GPT-4o 之后旗舰参数量大概率**没有继续单调上升**，而是配合 MoE / 蒸馏 / RL / tool-use 在「单位 token 智能密度」与「test-time compute」上下功夫。

---

## 3. 对齐技术演进：从 SFT 到 RLHF 到推理对齐

### 3.1 InstructGPT（2022）：RLHF 三步法的工程化定型

**论文**：Long Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*, **arXiv:2203.02155**, 2022-03-04。

InstructGPT 是 RLHF（Reinforcement Learning from Human Feedback）在大语言模型上的第一个**工业级、完整披露** 案例。其方法被业界归纳为「三步法」：

1. **SFT（Supervised Fine-Tuning）**：从 OpenAI API 真实用户 prompt 中采样，由约 40 名标注员撰写高质量回答（13K prompt-response 对）；在 GPT-3 175B 上做 16 epoch SFT。
2. **RM（Reward Model）训练**：标注员对同一 prompt 的 K=4~9 个候选回答做 *pairwise ranking*，得到约 33K 排序对；用 6B 模型（去掉 LM head 加 scalar head）做 ranking-loss 训练。
3. **PPO RL**：以 RM 输出为 reward，加 KL 惩罚（防 reward hacking + 模式坍塌），用 PPO 在 SFT 模型上更新策略；约 31K 用户 prompt 用作 RL 的 prompt 分布。

**核心结论**：1.3B 的 InstructGPT 在「人类偏好」上**胜过 175B 的 GPT-3**。这是「**对齐的杠杆远大于参数的杠杆**」首次被定量验证，直接刺激了整个行业的对齐投入。

**衍生设计**：
- KL constraint：$ R_{rl}(x, y) = R_\theta(x, y) - \beta \log \frac{\pi_{rl}(y|x)}{\pi_{sft}(y|x)} $
- *Mixing pretraining gradients*（PPO-ptx）：在 PPO 同步加入预训练 loss，以缓解公开 NLP benchmark 的「alignment tax」。

### 3.2 ChatGPT（2022-11）：对话 SFT × RLHF 的产品化

ChatGPT 是 InstructGPT 的对话版：在 InstructGPT 三步法基础上，
- **对话格式 SFT**：标注员**同时扮演 user 与 AI**，制造多轮对话数据；
- **system prompt** 引入：第一次明确把「**模型身份 / 安全策略 / 工具使用规则**」放进特殊 token 序列；
- **更激进的安全 RLHF**：把 safety 拒答与 helpfulness 一起放进奖励函数（具体函数未公开）。

ChatGPT 没有发表论文，OpenAI 仅在博客和 *Sparrow / Constitutional AI / RLHF 综述* 等同期工作中提及方法路径。其历史意义在产品维度（5 天破百万用户、全球现象级采用），不在算法维度。

### 3.3 GPT-4 时代的对齐升级：RBR、RLAIF 与 deliberative alignment

**资料**：GPT-4 System Card（2023）、*Rule-Based Rewards for Language Model Safety* (2024)、*Deliberative Alignment* (2024-12)。

GPT-4 / GPT-4o 时代对齐方法发生三方面升级：

1. **Rule-Based Rewards (RBR)**（2024，OpenAI 论文）：用 LLM-as-judge 把可程序化的安全规则（如「不应给出 bioweapon 配方」「应礼貌拒绝」）转成可微的 reward 信号，部分替换昂贵的人工 ranking；规则用自然语言写成 *Spec*，由 GPT-4 评分。
2. **RLAIF（RL from AI Feedback）**：在低风险标注上用 GPT-4 做 ranker，是 Anthropic Constitutional AI 之后的工业化版本；**OpenAI 未将整个 RLHF 替换为 RLAIF**，而是与人类反馈混合使用（*Implied*）。
3. **Deliberative Alignment（2024-12）**：用于 o 系列推理模型的对齐方法——在 RL 阶段，模型先 *reason about safety policy*（在 CoT 中显式引用 *OpenAI Safety Spec*）再决定 action。这是首次把「对齐」从静态约束变为「让模型自己 reason 出符合规范的回答」。论文 *Deliberative Alignment: Reasoning Enables Safer Language Models* (arXiv:2412.16339) 公开。

### 3.4 Model Spec 与 Spec-First 对齐

2024-05 OpenAI 公开《**Model Spec**》：用规范化文档（约 60 页，2025 年扩展至 ~100 页）显式定义模型应遵循的原则与排序规则（platform > developer > user > guideline）。Spec 既是 RBR 的输入，也是 deliberative alignment 中模型显式 reason 的对象。这种 *Spec-first* 设计与 Anthropic 的 Constitutional AI 殊途同归——把价值观从「隐性对齐」转为「显式可审计」。

### 3.5 对齐技术演进的逻辑链

```
GPT-3 raw LM
    └─ SFT (FeedME, text-davinci-002)
        └─ RLHF (PPO, text-davinci-003 / ChatGPT / InstructGPT)
            ├─ Conversation SFT + system prompt        ← ChatGPT
            ├─ Safety + helpfulness 双 reward          ← GPT-4
            ├─ Rule-Based Rewards (LLM-as-judge)       ← GPT-4o
            ├─ Process Reward / Outcome Reward         ← o1
            └─ Deliberative Alignment (CoT-on-Spec)    ← o1/o3
```

> 关键洞察：对齐范式正从「**结果对齐（output-level）**」→「**过程对齐（process-level）**」→「**规范对齐（spec-level reasoning）**」演化；而 o 系列推理模型让 CoT 本身成为对齐的承载介质。

---

## 4. 推理模型演进：o1 → o3 → o4-mini

### 4.1 o1（2024-09）：把 test-time compute 作为新的 scaling 维度

**资料**：OpenAI, *Learning to Reason with LLMs*, 2024-09-12（blog）、*OpenAI o1 System Card*, 2024-09-12。

o1（项目代号 *Strawberry / Q\**）首次在产品上把「思维链」从 prompt-time 技巧（CoT prompting / Self-Consistency）升级为**模型自带的、由 RL 训练出来的内部行为**。其要点：

- **CoT-as-policy**：模型在回答前生成大量内部 reasoning tokens（在 ChatGPT UI 中以「Thinking…」展示，但内容不暴露），最终输出 final answer。
- **RL on CoT**：用结果对错（math/code/science 这类可验证任务）作为奖励，让模型学会「写出**让自己更可能答对**的思维链」。OpenAI 官方图明确给出**两条 scaling 曲线**：train-time compute 与 **test-time compute** 都对 AIME pass@1 单调改善，且后者带来**新的可扩展轴**。
- **能力**：AIME 2024 单次采样 74%（GPT-4o 12%）、Codeforces Elo 1807（GPT-4o 808）、GPQA Diamond 78%（专业博士级理科）、IOI 2024 213/600（金牌区）。
- **显著弱项**：写作、对话、tool-use（首版 o1 不支持联网/代码执行）；ChatGPT-style 友好性较 GPT-4o 差。

**架构层面（推测，未确认）**：
- 与 GPT-4 / GPT-4o 共享某代基座，做了 **大规模 RL post-training**；
- *推测* 使用 *Process Reward Model（PRM）* 与 *Outcome Reward Model（ORM）* 结合，思路接近 OpenAI 此前论文 *Let's Verify Step by Step* (arXiv:2305.20050) 的 PRM800K 数据集；
- *推测* 没有完全外置的 search-tree（区别于 AlphaGo MCTS），更像 *implicit search via long CoT*。

### 4.2 o1-mini / o1-pro（2024-09 / 12）

- **o1-mini**：参数更小、专攻 STEM 推理（特别是数学和代码），价格降到 o1-preview 的 1/5；牺牲了通用知识广度（在 SimpleQA 等问答上明显差）。
- **o1-pro**：仅在 ChatGPT Pro $200/月套餐内提供，特点是**自动多次采样 + 自一致投票**（multi-sample best-of-N + ProSearch），在最难题上比单 o1 更稳。**相同模型，更高 test-time compute**——是 o 系列「计算即智能」论点的产品化。

### 4.3 o3 与 o3-mini（2024-12 / 2025-01）

**资料**：OpenAI, *Early access for safety testing: o3 and o3-mini*, 2024-12-20；*o3-mini System Card*, 2025-01。

o3 在 o1 公开三个月后亮相，定位是 **next generation reasoning model**：

- **基准跃迁**：
  - **ARC-AGI** 公共测试集 **87.5%**（high-compute setting，每题平均 ~1700× compute），首次跨过 François Chollet 设定的「人类基线」；low-compute 也有 **75.7%**。
  - **SWE-bench Verified 71.7%**（彼时第一）；
  - **EpochAI Frontier Math 25.2%**（被认为接近研究前沿数学难度）。
- **o3-mini**：2025-01-31 上线，分 low/medium/high 三档 *reasoning effort*；首次把 *reasoning effort* 暴露成 API 参数；medium 在 GPQA 上接近 o1，价格仅为 o1 的 ~7%。
- **2025-04 o3 全量上线 + 工具**：o3 在 ChatGPT 中支持 *python tool, search, image-input*，是首个能「**边推理边调工具**」的产品级推理模型。

**官方未披露**：参数量、是否仍走「RL on CoT」路线、是否引入 tree search 或 verifier 增强。社区根据 ARC-AGI 高 compute 模式的 1700× 推断 o3-pro 等价于 *N×自一致 + 后验 verifier*（*推测*）。

### 4.4 o4 / o4-mini（2025-04 起）

- **o4-mini**（2025-04-16）：以「reasoning at o3-mini cost」为口号，AIME 2024 97.4%（with tools）、Codeforces Elo 2719；首次在 ChatGPT 内默认开启 *python + search + image* 工具链。
- **o4**（旗舰版）截至 2025 年下半年仅在 ChatGPT Plus / Pro 内通过 GPT-5 Router 间接暴露，没有作为独立 API SKU 单独发布。
- 2025-Q4 起的 GPT-5 thinking 即为 o 系推理能力被合并入 GPT-5 router 的统一形态（详见 §2.9）。

### 4.5 推理模型范式的几个关键判断

1. **CoT-as-RL-policy 是新的 scaling 方向**：在 GPT-4 → GPT-4o 阶段「模型变大变快」红利收敛后，o1 把「同样的基座 + 更长的 RL post-training + 更长的 test-time CoT」证明为可持续的能力增长曲线。
2. **可验证任务驱动 RL 数据飞轮**：数学、代码、科学题、形式化证明这类任务的奖励是 *自动可计算* 的，从而能用 self-play / self-improvement 风格的循环训练出推理能力，**而不需要新的人工偏好数据**。
3. **CoT 同时承载推理与对齐**：deliberative alignment 把 *Spec* 写在 system prompt，模型在 CoT 中主动对照 Spec，这让对齐不再是「外置的 reward 形状」而是「内置的 reasoning 步骤」。
4. **推理模型与 chat 模型走向合流**：GPT-5 router 是这一合流的产品形态。预计 2026 年后旗舰模型不再区分 *chat* / *reasoning*，而是按 query 难度自适应分配 test-time compute。

---

## 5. 多模态演进：GPT-4V → GPT-4o → Realtime → Sora

### 5.1 GPT-4V（2023-09）：Image Encoder + LLM 的串联范式

**资料**：*GPT-4V(ision) System Card*, 2023-09-25。

GPT-4V 把 GPT-4 的视觉输入正式开放。**官方未披露架构**，但根据社区分析（*推测*）：
- 采用 *vision encoder (ViT-style) + projection MLP + LLM* 的串联结构（与 LLaVA / Flamingo 同代设计相近）；
- vision encoder 可能与 CLIP 系列共享谱系，但具体权重不公开；
- 视觉 token 与文本 token 在同一 sequence 中，由 LLM 自回归处理。

GPT-4V 主要场景：图像描述、文档/截图理解、UI 操作、医疗影像辅助（明确受限）等。OCR 上略弱于专门系统，但在「**理解图 + 推理**」上跨入新水位。

### 5.2 GPT-4o（2024-05）：Native Omnimodal

GPT-4o 是 OpenAI 第一个**真正端到端**的多模态模型，把 GPT-4V 的「外挂视觉」升级为「**统一 token 流**」：

- **音频原生化**：直接以离散音频 token（*推测* 类似 SoundStream / EnCodec 的 codec token）作为输入输出，端到端避免 ASR + TTS 串联的延迟与情感丢失；OpenAI 在 demo 中展示了模型「**听语调、回相应情绪**」的能力。
- **视觉原生化**：图像 / 视频帧通过 patch embedding 进入同一 sequence；对短视频（数秒）做时序理解。
- **能力副产品**：GPT-4o 是 OpenAI 第一个在英语外语种（中、阿、印、日等）上**显著优于** GPT-4 Turbo 的旗舰，因为 omnimodal token 化使得多语言文本 token 占用大幅下降（中文从 2.7×→1.4× 英文 token 数）。
- **Realtime API**（2024-10）：把 GPT-4o 的语音通道直接以 WebSocket 暴露，开发者可构建 ~300 ms 延迟的语音应用，是工业语音助手的拐点。

### 5.3 Sora（2024-02 → 2024-12）：视频生成

**资料**：OpenAI, *Video generation models as world simulators*, 2024-02-15；Sora 公测 2024-12-09。

虽然 Sora 不属于 GPT 主线（不是 LLM），但同代多模态战略上是关键：
- **Spacetime patches**：把视频切成时空 patch，作为视觉「token」喂给一个 *Diffusion Transformer (DiT)*；
- **统一控制接口**：以 GPT-4o 系生成的 caption 控制 Sora，使「文本 → 视频」过 LLM 这一段；
- **Video as world model**：OpenAI 在 blog 中明确把 Sora 框定为「**视频世界模型**」，是后续 GPT-5 多模态推理的前置实验场。

### 5.4 GPT-5 多模态（2025-08）

GPT-5 在多模态评测上：
- MMMU 84.2、MMMU-Pro 78.4，是当时旗舰最高水准（2025 年 8 月）；
- 视频理解方面 OpenAI 仅在 demo 与博客中演示，没有给出 standardized benchmark；
- 在 ChatGPT 内集成 Sora（独立产品）、Advanced Voice Mode（GPT-4o realtime 路径继承）、Image Generation（基于 4o image generation，2025-03 升级为 GPT-Image-1）。

### 5.5 多模态架构演进总结

| 阶段 | 代表 | 视觉 | 音频 | 视频 | 工程模式 |
|---|---|---|---|---|---|
| GPT-4V (2023.09) | GPT-4 | 编码器外挂 → LLM | 无（Whisper 串联） | 无 | 单向：image→token→LLM |
| GPT-4o (2024.05) | GPT-4o | 原生 patch token | 原生音频 token | 短帧序列 | 统一 token 流，端到端 |
| Sora (2024.12) | Sora DiT | spacetime patches 反向生成 | 无 | 文 → 视频生成 | DiT，独立模型 |
| GPT-5 (2025.08) | GPT-5 router | 同 4o + 推理增强 | 同 4o realtime | 演示级 | 多模态 + reasoning 合流 |

> 一个重要观察：**GPT-4o 之后，OpenAI 已不再发表独立的「多模态架构论文」**。一切技术都走 System Card 路径，结合产品发布节奏；这与 Google 在 Gemini / Anthropic 在 Claude 上的策略类似——架构成为商业秘密，能力以评测和产品形态对外。

---

## 6. 跨代技术演进分析

### 6.1 七条主线的演进图

```
1) 架构主线：       Decoder Transformer (GPT-1) → Pre-LN + 大词表 (GPT-2) → 175B Dense (GPT-3) → MoE? (GPT-4*) → Omnimodal (GPT-4o) → Router (GPT-5)
2) 数据主线：       BookCorpus → WebText → CC + Books → 多模态 + 代码 + 合成数据
3) 计算主线：       2.6e19 → 1.5e21 → 3.14e23 → ~2e25*  → ?  (* 指 GPT-4 的推测算力)
4) 适配主线：       Fine-tuning (GPT-1) → Zero-shot (GPT-2) → Few-shot ICL (GPT-3) → SFT+RLHF (ChatGPT) → RL on CoT (o1)
5) 多模态主线：     纯文本 → CLIP/DALL·E 分支 → GPT-4V 串联 → GPT-4o native → Sora world model → GPT-5 multimodal reasoning
6) 对齐主线：       无 → 不开源治理 (GPT-2) → SFT (FeedME) → RLHF → RBR/RLAIF → Deliberative Alignment
7) 商业主线：       论文+权重 → 论文+API → 仅 API → 闭源 + System Card + Router 套餐
```

### 6.2 三次范式跃迁的判断

1. **Pre-train-and-prompt 取代 Pre-train-and-fine-tune**（2019-2020）：GPT-2/3 用 ICL 把 NLP 从「每任务一个模型」变成「一个模型应对所有任务」。
2. **RLHF 取代 SFT 作为产品对齐方法**（2022）：InstructGPT/ChatGPT 证明对齐对最终用户效用的杠杆远大于参数；自此所有竞品（Claude、Gemini、LLaMA-Chat、Qwen-Instruct）都把 RLHF/DPO 系列方法视为标配。
3. **Test-time compute 作为新的 scaling 轴**（2024-2025）：o1 把推理时长变成可投入资源；这使「同尺寸模型 + 更多思考时间」可以解决以前需要更大模型才能解决的问题，给行业打开第二增长曲线。

### 6.3 OpenAI 公开节奏的演化

| 年份 | 公开级别 | 代表 |
|---|---|---|
| 2018 | 完整论文 + 权重 | GPT-1 |
| 2019 | 完整论文 + 阶段权重（受治理争议） | GPT-2 |
| 2020 | 完整论文 + 仅 API | GPT-3 |
| 2022 | 完整论文（仅对齐方法）+ 仅 API | InstructGPT |
| 2023+ | Tech Report / System Card，无架构与数据 | GPT-4 / 4o / o1 / GPT-5 |
| 2025+ | 仅 changelog + System Card + 评测 | GPT-4.1 / GPT-5 子版本 |

> 这一趋势既是商业护城河选择，也对学术复现造成了真实障碍。GPT-3 是最后一个「学界可深度复现路径」的 OpenAI 旗舰；之后行业实际复现工作转移到 Meta (LLaMA)、Mistral、Qwen、DeepSeek 等开源阵营。

### 6.4 OpenAI 与开源阵营的相对位置

- **基座规模**：开源端的 LLaMA-3 405B、DeepSeek-V3 671B、Qwen3-235B 等已在公开权重上接近 GPT-4 级，**但 OpenAI 旗舰仍领先于「智能密度」与「指令/工具一致性」**（评测口径有争议）。
- **推理模型**：DeepSeek-R1（2025-01）、Qwen3-Thinking、Kimi k1.5 都用类似 RL on CoT 路径追上 o1；o3/o4 在 ARC-AGI 等极硬任务上仍领先。
- **多模态**：开源阵营追赶最快的环节；Qwen3-Omni / Kimi-VL-Thinking 在视觉理解上接近 GPT-4o，但 realtime audio 与 video 生成（Sora 级）仍保持代差。

---

## 7. 关键创新点（按行业影响力排序）

> 排序基于「对全行业范式的重塑程度」，而非单点性能。

### 7.1 In-Context Few-Shot Learning（GPT-3, 2020）

- **影响力级别**：★★★★★（重塑 NLP 范式）
- **贡献**：首次在工业规模上证明「prompt 即学习」，使 LLM 成为通用接口。所有后续 instruction-tuning、RLHF、Agent 框架都建立在 ICL 之上。
- **学术地位**：NeurIPS 2020 Best Paper；改变了 NLP 学界的研究语言。

### 7.2 RLHF 三步法（InstructGPT/ChatGPT, 2022）

- **影响力级别**：★★★★★（让 LLM 可商用化）
- **贡献**：把对齐做成工程流水线（SFT → RM → PPO），引爆 ChatGPT 现象；之后 *人人都做 RLHF*。
- **副作用**：催生整个 RLHF 生态（DPO、IPO、KTO、ORPO、GRPO 等改进算法）。

### 7.3 Test-time Compute Scaling（o1, 2024）

- **影响力级别**：★★★★★（开辟第二增长曲线）
- **贡献**：在「数据/参数 scaling 接近瓶颈」的舆论中，给出 *test-time scaling* 这条新维度，让 LLM 能力增长继续可预测。
- **派生**：deliberative alignment、verifier-augmented decoding、tree-of-thought 全面被推上工业落地。

### 7.4 Native Omnimodal Token Stream（GPT-4o, 2024）

- **影响力级别**：★★★★☆
- **贡献**：以单模型端到端处理 text/audio/image/video，把语音助手延迟从秒级压到 ~300 ms；成为后续多模态原生模型（Gemini 2.x、Qwen-Omni 等）的标杆。

### 7.5 Predictive Scaling Laws（GPT-4 Tech Report, 2023）

- **影响力级别**：★★★★☆
- **贡献**：把训练损失从 1000× 较小算力外推预测，是 *工程化 scaling law* 的首次公开例证；让大型训练任务的预算与风险可估算。

### 7.6 Decoder-only Transformer + LM 预训练（GPT-1, 2018）

- **影响力级别**：★★★★☆
- **贡献**：奠定了 *Decoder-only + autoregressive LM* 这一今日所有大模型默认的架构范式。

### 7.7 Spec-First Alignment / Deliberative Alignment（2024）

- **影响力级别**：★★★☆☆
- **贡献**：把对齐从「奖励函数形状」转为「显式的 Spec + 模型自我推理」，使对齐过程可审计、可演进。

### 7.8 Reasoning Effort 作为 API 一等公民（o3-mini, 2025）

- **影响力级别**：★★★☆☆
- **贡献**：第一次把「思考多久」做成开发者旋钮（low/medium/high），使「推理预算」与「token 预算」并列为产品能力。

### 7.9 GPT-5 Router：模型形态合流（2025）

- **影响力级别**：★★★☆☆（待观察）
- **贡献**：把 chat 模型与 reasoning 模型合并为一个产品入口，预示未来旗舰将以「模型集群 + 路由」为基本单位，而非单一权重。

### 7.10 长上下文 + JSON Mode + Function Calling（GPT-4 Turbo, 2023）

- **影响力级别**：★★☆☆☆（产品级标准化）
- **贡献**：把 LangChain 等中间层标准化进 API，是 LLM-as-OS 概念的工程化先声。

---

## 8. 参考文献

> 引用规则：论文优先标注 arXiv 编号；OpenAI 博客与 System Card 仅给标题与日期，可据此搜索。

### 8.1 OpenAI 一手论文 / 报告

1. Radford, A. et al. *Improving Language Understanding by Generative Pre-Training* (GPT-1). OpenAI Tech Report, 2018-06-11.
2. Radford, A. et al. *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI Tech Report, 2019-02-14.
3. Brown, T. B. et al. *Language Models are Few-Shot Learners* (GPT-3). **arXiv:2005.14165**, NeurIPS 2020.
4. Chen, M. et al. *Evaluating Large Language Models Trained on Code* (Codex). **arXiv:2107.03374**, 2021-07-07.
5. Ouyang, L. et al. *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT). **arXiv:2203.02155**, 2022-03-04.
6. OpenAI. *GPT-4 Technical Report*. **arXiv:2303.08774**, 2023-03-14.
7. OpenAI. *GPT-4V(ision) System Card*. 2023-09-25.
8. OpenAI. *GPT-4o System Card*. 2024-08-08.
9. Lightman, H. et al. *Let's Verify Step by Step* (PRM800K). **arXiv:2305.20050**, 2023-05-31.
10. OpenAI. *Learning to Reason with LLMs* (o1 blog). 2024-09-12.
11. OpenAI. *OpenAI o1 System Card*. 2024-09-12.
12. Guan, M. et al. *Deliberative Alignment: Reasoning Enables Safer Language Models*. **arXiv:2412.16339**, 2024-12-20.
13. OpenAI. *o3 / o3-mini Early Access for Safety Testing*. 2024-12-20 blog.
14. OpenAI. *o3-mini System Card*. 2025-01-31.
15. OpenAI. *Introducing GPT-4.1 in the API*. 2025-04-14 blog.
16. OpenAI. *Introducing OpenAI o3 and o4-mini*. 2025-04-16 blog.
17. OpenAI. *GPT-5 System Card*. 2025-08-07.
18. OpenAI. *Introducing GPT-5*. 2025-08-07 blog.
19. OpenAI. *Sora: Video generation models as world simulators*. 2024-02-15.
20. OpenAI. *Model Spec*. 2024-05（持续更新）。
21. Mu, T. et al. *Rule-Based Rewards for Language Model Safety*. OpenAI 2024.

### 8.2 直接相关 OpenAI / 学术工作

22. Vaswani, A. et al. *Attention Is All You Need*. **arXiv:1706.03762**, 2017.
23. Kaplan, J. et al. *Scaling Laws for Neural Language Models*. **arXiv:2001.08361**, 2020.
24. Hoffmann, J. et al. *Training Compute-Optimal Large Language Models* (Chinchilla). **arXiv:2203.15556**, 2022.
25. Wei, J. et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. **arXiv:2201.11903**, 2022.
26. Wei, J. et al. *Emergent Abilities of Large Language Models*. **arXiv:2206.07682**, 2022.
27. Bai, Y. et al. (Anthropic). *Constitutional AI: Harmlessness from AI Feedback*. **arXiv:2212.08073**, 2022.
28. Snell, C. et al. *Scaling LLM Test-Time Compute Optimally...*. **arXiv:2408.03314**, 2024.
29. DeepSeek-AI. *DeepSeek-R1*. **arXiv:2501.12948**, 2025-01.
30. Chollet, F. *On the Measure of Intelligence* (ARC). **arXiv:1911.01547**, 2019.

### 8.3 第三方分析 / 产业分析（推测来源，仅作背景）

31. SemiAnalysis (Dylan Patel) 关于 GPT-4 的 MoE 架构分析（博客文章，2023-07）。
32. George Hotz (geohot) 关于 GPT-4 8×220B 的公开陈述（2023 年播客）。
33. Epoch AI Frontier Math 评测、ARC-AGI 公共榜单（2024-2025）。
34. SWE-bench Verified、Aider Polyglot、GPQA Diamond 等学术 / 社区评测榜单（2023-2025）。

---

> **声明**：本报告所有 *推测* 内容（GPT-4 的参数 / MoE 配置、训练成本、GPT-4o 与 GPT-5 的架构细节等）均来源于学界与产业界的第三方分析，未被 OpenAI 任何官方文件确认。读者引用时请保留「推测」标签，以免误用。
