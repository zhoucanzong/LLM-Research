# Meta LLaMA 系列大模型完整研究报告

> 调研范围：LLaMA 1 → LLaMA 2 → Code Llama → LLaMA 3 → LLaMA 3.1 / 3.2 / 3.3 → LLaMA 4（Scout / Maverick / Behemoth）→ 2025-2026 最新进展
>
> 主线视角：架构演进（Dense Transformer → GQA → MoE）、训练规模跃升（1.4T → 2T → 15T+ tokens）、开源策略递进、对齐范式（RLHF → DPO）、多模态扩展、对开源生态的辐射效应。

---

## 目录

1. [概述与时间线](#1-概述与时间线)
2. [基座模型演进](#2-基座模型演进)
3. [多模态能力演进](#3-多模态能力演进)
4. [代码模型：Code Llama](#4-代码模型code-llama)
5. [跨代技术演进分析](#5-跨代技术演进分析)
6. [开源生态影响](#6-开源生态影响)
7. [关键创新点](#7-关键创新点)
8. [2025-2026 最新动态与展望](#8-2025-2026-最新动态与展望)
9. [参考文献](#9-参考文献)

---

## 1. 概述与时间线

LLaMA（Large Language Model Meta AI）是 Meta（前 Facebook AI Research / GenAI 团队）自 2023 年 2 月推出的开源 / 开放权重大语言模型系列。它在两年多时间内连续放出至少 9 个主要里程碑版本，从纯文本研究模型一路演进至 native multimodal、稀疏 MoE、千万级长上下文的旗舰级模型，是当前开源 LLM 生态毫无疑问的"事实标准"之一。其重要性体现在两条主线：

- **技术主线**：从 7B-65B Dense Transformer，扩展到 405B Dense"巨无霸"，再到 17B 激活 / 128 expert 的稀疏 MoE，并自 LLaMA 3.2 起原生支持视觉模态、自 LLaMA 4 起原生支持多模态早期融合（early-fusion）。
- **生态主线**：LLaMA 1 仅供研究；LLaMA 2 起开放商用（带条件许可）；LLaMA 3 进一步放宽多语言与商用边界；LLaMA 4 仍延续 Community License。其衍生家族（Alpaca、Vicuna、Guanaco、WizardLM、Code Llama、LLaVA、Llama-Adapter、QLoRA 等）几乎重塑了 2023-2025 年开源 LLM 与 PEFT 生态的研究范式。

### 1.1 时间线（按发布时间排序）

| 时间 | 模型 | 规格 | 核心亮点 |
| --- | --- | --- | --- |
| 2023-02 | **LLaMA 1** | 7B / 13B / 33B / 65B；2K context；1.0–1.4T tokens | RMSNorm 预归一化、RoPE、SwiGLU；研究专用许可 |
| 2023-07 | **LLaMA 2** | 7B / 13B / 70B；4K context；2T tokens | GQA（仅 70B）、SFT+RLHF（PPO+rejection sampling）、Ghost Attention；开放商用 |
| 2023-08 | **Code Llama** | 7B / 13B / 34B（后增 70B）；16K → 100K context | 基于 LLaMA 2 二次预训练（500B 代码 tokens）+ FIM；Python 专精与 Instruct 变体 |
| 2024-04 | **LLaMA 3** | 8B / 70B；8K context；15T tokens | 全规模 GQA、128K 词表 tiktoken-style 分词器、扩展高质量数据 |
| 2024-07 | **LLaMA 3.1** | 8B / 70B / **405B**；128K context；15.6T tokens | 旗舰 405B Dense、长上下文、多语言、SFT+rejection sampling+DPO 多轮迭代 |
| 2024-09 | **LLaMA 3.2** | 1B / 3B（轻量）；11B / 90B（视觉） | 首次原生视觉模型（cross-attention 适配器）；端侧轻量模型蒸馏自 8B/70B |
| 2024-12 | **LLaMA 3.3** | 70B Instruct | 仅文本对话版；以 70B 体量逼近 3.1 405B 性能 |
| 2025-04 | **LLaMA 4 Scout / Maverick** | 17B 激活；16 / 128 experts；10M context | 首次 MoE、native multimodal（early fusion）、iRoPE 长上下文方案 |
| 2025-04+ | **LLaMA 4 Behemoth** | 288B 激活 / ~2T 总参数 | 教师模型（仍在训练）；用于蒸馏 Scout / Maverick |
| 2025-2026 | 持续增量 | — | Meta 强调"用 LLaMA 4 蒸馏 / 应用"为主，配合 PEFT、长上下文、Agent 等生态 |

### 1.2 报告导读

本报告以"基座 → 专项 → 跨代分析 → 生态影响"为骨架，重点回答以下问题：

1. 从 LLaMA 1 到 LLaMA 4，**架构差异**到底是什么？为什么 GQA、RoPE、SwiGLU 在每一代都被保留？为什么到 LLaMA 4 才转向 MoE？
2. 训练规模为何从 1.4T → 2T → 15T tokens **三级跳**？背后的数据策略是什么？
3. 对齐范式如何从 RLHF（PPO）演进到 **rejection sampling + DPO** 多轮迭代？
4. LLaMA 3.2 / 4 的多模态架构选择有何差别？
5. LLaMA 4 的 MoE 与 DeepSeek V2/V3 的 MoE 在专家粒度、共享专家、路由策略上有何异同？
6. LLaMA 系列对 Alpaca / Vicuna / LLaVA / QLoRA 等开源生态的辐射作用如何？

---

## 2. 基座模型演进

### 2.1 LLaMA 1（2023-02）：奠定 LLaMA 架构范式

LLaMA 1 是 Meta 第一个真正"出圈"的 LLM 系列。其核心论文 *"LLaMA: Open and Efficient Foundation Language Models"*（Touvron et al., arXiv:2302.13971）发布了 7B / 13B / 33B / 65B 四个尺寸的纯文本 Decoder-only Transformer。关键特征：

- **数据规模**：7B / 13B 模型训练 1.0T tokens；33B / 65B 训练 1.4T tokens。数据全部来自公开语料：CommonCrawl（67%）、C4（15%）、GitHub、Wikipedia、Books、ArXiv、StackExchange 等。LLaMA 1 明确选择"超过 Chinchilla optimal"的训练规模，目的是让小模型在推理时也具备竞争力。
- **架构修改**（相对原始 Transformer 的三处关键改动，构成此后 LLaMA 全系列的"DNA"）：
  1. **Pre-normalization with RMSNorm**：每个子层的输入而非输出做归一化；归一化方式从 LayerNorm 换为 RMSNorm（Zhang & Sennrich, 2019），减少均值计算并提升训练稳定性。
  2. **SwiGLU**：FFN 激活换成 Shazeer 在 PaLM 中使用的 SwiGLU（Swish-Gated Linear Unit），中间维度按 (2/3)·4d 取整以保持参数量。
  3. **Rotary Positional Embedding (RoPE)**：用 Su et al. 2021 提出的旋转位置编码替换原始 absolute positional embedding，为后续长上下文扩展奠定基础。
- **优化器与训练**：AdamW（β1=0.9, β2=0.95），cosine LR schedule，2048 上下文窗口，batch size 4M tokens 量级；65B 在 2048 张 A100-80GB 上训练 21 天。
- **能力表现**：13B 在多数 benchmark 上超过 GPT-3 175B；65B 与 Chinchilla-70B / PaLM-540B 同档。
- **许可**：研究专用，需向 Meta 申请并签署非商用协议；权重一周内"意外泄漏"到 4chan/HuggingFace，反而催生了空前的开源 LLM 复刻浪潮。

### 2.2 LLaMA 2（2023-07）：开放商用与对齐范式落地

LLaMA 2（Touvron et al., *"Llama 2: Open Foundation and Fine-Tuned Chat Models"*, arXiv:2307.09288）首次将 LLaMA 作为可商用的"开放权重"产品发布，是 LLaMA 系列的第一个分水岭。

- **尺寸与数据**：放出 7B / 13B / 70B（取消 33B；34B 内部存在但未发布）；预训练数据扩大至 **2T tokens**（约为 LLaMA 1 的 1.4–2 倍），公开数据，过滤了已知含大量个人信息的网站；上下文从 2K **扩展至 4K**。
- **架构改动**：
  - 总体仍是 LLaMA 1 同款 Pre-RMSNorm + RoPE + SwiGLU；
  - **首次引入 Grouped-Query Attention (GQA)**，但**仅在 70B 上启用**（KV head=8）。这是 LLaMA 系列长期最关键的工程优化之一，显著降低 KV cache 大小、加速长文本推理。
- **对齐管线**（首次工业级开源 RLHF 全流程）：
  - **SFT**：高质量人工写作的指令数据 ~28K；
  - **奖励模型**：分别训练 Helpfulness RM 与 Safety RM 两个独立模型；
  - **RLHF**：Rejection Sampling fine-tuning（5 轮）+ PPO，组合使用而非纯 PPO；
  - **Ghost Attention（GAtt）**：一种数据增强技巧，通过把 system prompt 注入到所有用户回合，再仅对最后一回合保留，让模型在多轮对话中"记住"角色或约束。
- **安全**：发布 Safety Reward Model、Red Teaming 报告以及 *Responsible Use Guide*，明确禁止用于军事、儿童不当内容等用途。
- **许可**：Llama 2 Community License，**允许商用**（月活 7 亿以上的公司需向 Meta 申请例外）。这一步直接打开了企业落地的闸门。

### 2.3 LLaMA 3（2024-04）：tokenizer / 数据 / GQA 三重升级

LLaMA 3（Meta, 2024-04 博客；详细论文见 *"The Llama 3 Herd of Models"*）发布 8B / 70B 两个尺寸，与 LLaMA 2 相比是数据驱动的"升级版本"。

- **训练数据**：从 2T 跃升到 **15T+ tokens**（约 7 倍 LLaMA 2），全部公开来源，比 LLaMA 2 多出 4× 的代码、>30 种语言的非英文数据（占比约 5%，仍以英文为主）；并通过分类器和"质量过滤启发式"严格筛选高质量子集。
- **Tokenizer 升级**：从 LLaMA 2 的 32K SentencePiece BPE 词表，扩展到 **128K tiktoken 风格 BPE**，编码效率显著提升（每 token 表达更多信息），尤其改善多语言与代码场景。
- **架构**：
  - 8B 与 70B 均启用 GQA（首次将 GQA 下沉到所有尺寸）；
  - 上下文窗口在原始版本中为 8K；
  - 训练语料长度配合 8K 序列。
- **对齐**：SFT + Rejection Sampling + PPO + DPO 组合。Meta 开始在 LLaMA 3 阶段把 **DPO** 引入工业级管线，并用 PPO 与 DPO 并行验证。

### 2.4 LLaMA 3.1（2024-07）：405B 旗舰与长上下文

*"The Llama 3 Herd of Models"*（arXiv:2407.21783）作为完整技术报告发布的同时，Meta 推出 8B / 70B / **405B** 三尺寸的 LLaMA 3.1：

- **旗舰 405B Dense**：全球当时**最大、最强的开放权重 Dense 模型**，FP16 体量约 800 GB；论文明确选择 Dense 而非 MoE，原因是"Dense 在推理与对齐工程上更可控"，与 DeepSeek、Mixtral、Qwen 等同期路线形成鲜明对比。
- **长上下文**：上下文从 8K 扩展到 **128K**。技术手段：
  - 增大 RoPE base frequency 至 **500,000**；
  - 多阶段长上下文继续预训练（Continued Pretraining on Long Documents），逐步从 8K → 16K → 32K → 64K → 128K；
- **训练规模**：405B 在 ~16K H100 GPU 上以 BF16 训练 ~54 天，~3.8×10²⁵ FLOPs；累计 15.6T tokens。
- **对齐管线（一图看懂）**：6 轮 **SFT → Rejection Sampling → DPO** 迭代，每轮把最新模型作为标注/拒绝采样的 backbone，随后蒸馏到 8B / 70B（**用 405B 给 8B/70B 当老师** 是 LLaMA 3.1 的另一关键方法学）。
- **多语言**：8 种核心语言（英、德、法、意、葡、印地、西、泰）。
- **生态意义**：405B 模型权重开源，配套 Llama Stack 工具链；与 GPT-4o / Claude 3.5 Sonnet 在 MMLU、GSM8K、HumanEval、IFEval 等多数基准上同档。

### 2.5 LLaMA 3.2（2024-09）：多模态 + 端侧轻量两手抓

LLaMA 3.2（2024-09 Meta Connect 大会发布）形成两条产品线：

- **端侧轻量**：1B、3B 纯文本，128K 上下文。这两个模型通过 **结构化剪枝 + 知识蒸馏**（来自 8B / 70B）训练得到，专为手机、PC、边缘设备优化（已在高通 / 联发科平台验证）。
- **视觉多模态**：11B-Vision、90B-Vision；视觉部分采用 **额外训练的 Vision Adapter**：
  - 视觉编码器（ViT 风格，对比学习预训练）；
  - 通过 **cross-attention 层** 注入到原 Llama 3.1 文本模型；
  - 文本骨干**冻结**，只训练新增的 cross-attention 与 adapter，从而保留 3.1 的纯文本能力；
- **意义**：完成了 LLaMA 系列"从 LLM 到 LMM"的转身，但仍属于**晚融合（late-fusion adapter）**架构；后续 LLaMA 4 才走向真正 native multimodal。

### 2.6 LLaMA 3.3（2024-12）：70B 体量逼近 405B 智能

LLaMA 3.3 70B Instruct（2024-12-06 发布）是个"小而精"的过渡版本：

- 仅 70B 单尺寸、Text-only、Instruct 对话版；
- 通过升级的 post-training（对齐数据扩展 + 多轮 DPO）把 70B 的指令遵循、推理与多语言能力提升至**与 LLaMA 3.1 405B 相当**；
- 推理成本仅为 405B 的几分之一，迅速成为 2024-Q4 至 2025-Q1 期间企业落地最广的 LLaMA 版本之一；
- 许可：Llama 3.3 Community License（沿用 3.1 风格）。

### 2.7 LLaMA 4（2025-04）：MoE + Native Multimodal + 10M 长上下文

LLaMA 4（Meta AI 博客 *"The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation"*, 2025-04）是 LLaMA 系列的第二个分水岭：**首次切换到稀疏 MoE 架构**、**首次原生多模态**、**首次跨入十百万级 token 长上下文**。

发布的"四模型 herd"：

| 模型 | 激活参数 | 总参数（MoE 总和） | 专家数（routed + shared） | 上下文 | 模态 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| **LLaMA 4 Scout** | 17B | ~109B | 16 routed + 1 shared | **10M tokens** | Text + Image（native） | 已开放权重 |
| **LLaMA 4 Maverick** | 17B | ~400B | 128 routed + 1 shared | 1M（部分文档称 10M） | Text + Image（native） | 已开放权重 |
| **LLaMA 4 Behemoth** | 288B | ~2T | 16 experts（粒度更大） | ~10M | Text + Image | 仍在训练，作为教师模型 |
| LLaMA 4 Reasoning（路标） | — | — | — | — | — | 路标产品，未独立发布 |

#### 2.7.1 MoE 架构关键点

- **共享专家（shared expert）**：每层 Transformer FFN 由 1 个 shared expert + N 个 routed experts 组成。每个 token 总是经过 shared expert，再通过 router 选 1 个 routed expert（top-1）。这种"1 shared + 1 routed"组合在工程上等同于一种"Conditional + Always-on 复合 FFN"。
- **路由**：top-1 token-choice routing；与 DeepSeek V2/V3 的 fine-grained + top-K routing 路线**有意做出区分**。
- **专家粒度**：Maverick 的 128 个 routed expert 比 Scout 的 16 个细粒度许多；Behemoth 仅 16 个但每个体量巨大。Meta 用"专家数 vs 专家大小"两端探索 MoE 设计空间。
- **MoE 与 Dense 层交错**：参考 Switch Transformer 经验，并非每一层都是 MoE，部分层保留 Dense FFN 作稳定锚点。
- **iRoPE（interleaved RoPE）**：为支撑 10M 上下文，Scout 引入 *interleaved Rotary Position Embeddings* + 推理期 attention temperature scaling 的组合方案。部分注意力层使用 NoPE（无位置编码），让"无位置"层为长程外推提供"位置无关"语义通道。
- **Native Multimodal (Early Fusion)**：与 LLaMA 3.2 的 cross-attention adapter 不同，LLaMA 4 在 token 序列层面把图像 token 与文本 token 拼接送入同一 Transformer 主干（**early fusion**）；图像编码器使用基于 MetaCLIP 的改进版本。
- **训练规模**：Scout 训练 ~40T tokens，Maverick ~22T tokens，Behemoth ~30T tokens（混合文本 + 图像 + 视频帧）；后训练 pipeline 用 Behemoth 作为教师，针对 Maverick 进行**联合蒸馏 loss**预训练。
- **后训练**：完全重写的 *轻量 SFT → online RL → 轻量 DPO* 管线，以避免对推理路径的过度规整化（"避免 over-training on safety/format"）。

#### 2.7.2 与 DeepSeek MoE 的对比

| 维度 | LLaMA 4（Scout / Maverick） | DeepSeek-V2 / V3 / R1 |
| --- | --- | --- |
| 路由 | Top-1 token-choice | Top-K（V3：1 shared + top-8 routed），细粒度 |
| 共享专家 | 1 个 | 1 个（V3） |
| 专家数 | 16 / 128 | 256（V3 routed），细粒度专家 |
| 激活 / 总参数 | 17B / 109B 或 400B | 21B / 236B（V2）；37B / 671B（V3） |
| 多模态 | Native early fusion | 主要文本（DeepSeek-VL 系列单独路线） |
| 长上下文方案 | iRoPE + NoPE 交错 + temp scaling | YaRN-style（V2/V3） |
| 训练 token | 22–40T | V3 ~14.8T |
| 蒸馏教师 | Behemoth（~2T） | R1 → V3 / Lite |

可以看出 LLaMA 4 走"少量大专家 + 共享专家 + 极长上下文 + 原生多模态"路线，DeepSeek 走"大量细粒度专家 + 极致 MFU + 强推理强化学习"路线，两者形成 2025 年开源 MoE 的两个主流分支。

---

## 3. 多模态能力演进

LLaMA 系列的多模态化经历了三个阶段，反映了开源 LMM 的整体范式迁移：

### 3.1 LLaMA 1/2 时代（外部社区驱动）

LLaMA 1/2 自身是纯文本模型，但社区基于 LLaMA 构建了一批早期 VLM：
- **LLaVA**（Liu et al., 2023）：CLIP ViT-L 视觉编码器 + 简单 MLP projector + Vicuna（LLaMA 衍生）作语言模型，开创了 *visual instruction tuning* 范式。
- **MiniGPT-4 / Otter / mPLUG-Owl** 等：均以 LLaMA 1/2 为底座，验证了"冻结 LLM + 可训练 projector"是一条便宜且强力的路线。

### 3.2 LLaMA 3.2 阶段（官方 Cross-Attention Adapter，late fusion）

LLaMA 3.2 11B/90B Vision 是 Meta 第一个**官方多模态**模型：

- 视觉塔：ViT 风格图像编码器，独立预训练于大规模图文对（contrastive + 后续视觉指令数据）；
- 桥接：在 LLaMA 3.1 8B / 70B 文本骨干中插入 **cross-attention 层**，每隔若干 self-attention 层注入一次图像 KV，类似 Flamingo 的 cross-attention adapter；
- 训练：**冻结文本主干**，仅训练 cross-attention 与适配模块；这保证了模型的纯文本能力 100% 与 3.1 一致，同时获得视觉能力。这种"晚融合 / 适配"路线工程风险低，但限制了图像与文本表征的深度融合。

### 3.3 LLaMA 4 阶段（Native Early Fusion）

LLaMA 4 Scout / Maverick 是首批 **natively multimodal** 的 LLaMA：

- 视觉 / 文本 / 视频帧都被编码为 token，**早期就拼接进同一个 Transformer 主干**；
- 一次性预训练（不再"先文本后图像"两阶段）；
- 视觉 encoder 改进自 MetaCLIP，配合 patch tokenization；
- 配合 10M 长上下文，足以一次性处理整本 PDF + 视频帧序列；
- 推理时图像 token 与文本 token 共享同一注意力机制，没有 cross-attention 单独通道。

这一切换标志着 LLaMA 系列从"LLM + 视觉适配器"演变为"LMM 原生模型"，与 Gemini、GPT-4o、Qwen2.5-VL 的方向一致。

---

## 4. 代码模型：Code Llama

Code Llama（Rozière et al., *"Code Llama: Open Foundation Models for Code"*, arXiv:2308.12950）于 2023-08 发布，是基于 LLaMA 2 二次预训练得到的代码专精家族。

### 4.1 训练管线

- **基础模型**：LLaMA 2（7B / 13B / 34B；后增 70B）；
- **代码二次预训练**：在 LLaMA 2 之上额外训练 **500B tokens** 的代码数据（GitHub、StackExchange 代码相关、自然语言代码混合）；
- **长上下文阶段**：所有变体在 16K 序列长度上训练，并在更长（最长 100K）输入上保持稳定；通过 RoPE base frequency 调高（θ=1e6）实现外推；
- **三个变体**：
  - **Code Llama**（基座）：通用代码模型；
  - **Code Llama – Python**：再额外训练 100B Python 代码；
  - **Code Llama – Instruct**：在自然语言 + 代码 instruction-following 数据上 SFT，用于对话式编程助手；
- **70B 版本**（2024-01 追加发布）：训练 1T tokens，效果明显优于 34B；HumanEval pass@1 从 34B 的 ~48% 提升至 ~67%。

### 4.2 关键技术

- **Fill-in-the-Middle (FIM) / Infilling**：7B / 13B 训练时引入 FIM 目标（前缀 + 后缀 → 中间），使其能对已有代码做插入式补全（IDE 场景核心能力）；34B 不做 FIM，专注 next-token，以最大化生成质量。
- **长上下文外推**：通过 RoPE θ 调整，使模型在 100K 输入上仍稳定工作，是开源代码模型最早突破 32K 的工作之一。
- **Instruct 训练数据**：使用 self-instruct 风格生成的"代码问答"对，配合 LLaMA 2 Chat 安全数据，避免代码助手生成恶意内容。

### 4.3 影响

Code Llama 推动了 WizardCoder、DeepSeek Coder、StarCoder2、CodeQwen 等后续代码 LLM 的范式（"通用基座 + 代码 continued pretraining + 代码 SFT"），并直接成为 Meta 自家 GitHub Copilot 替代项目（如 Llama Code Assistant 等）的底座。后续 LLaMA 3 时代，Meta 不再单独发布 Code Llama 3，而是把代码能力直接合入 LLaMA 3 / 3.1 / 3.3 主干（代码数据占预训练混合中显著比例），这也是模型通用化趋势的体现。

---

## 5. 跨代技术演进分析

### 5.1 架构演进（Dense → GQA → MoE）

| 维度 | LLaMA 1 (2023.02) | LLaMA 2 (2023.07) | LLaMA 3 (2024.04) | LLaMA 3.1 (2024.07) | LLaMA 3.2 (2024.09) | LLaMA 3.3 (2024.12) | LLaMA 4 (2025.04) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 架构类型 | Dense | Dense | Dense | Dense | Dense + Vision Adapter | Dense | **Sparse MoE + Vision (early-fusion)** |
| 归一化 | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| 激活 | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| 位置编码 | RoPE (θ=1e4) | RoPE (θ=1e4) | RoPE | RoPE (θ=5e5)，长上下文 | RoPE (θ=5e5) | RoPE | **iRoPE + NoPE 交错** |
| 注意力 | MHA | MHA（7B/13B），**GQA（70B）** | **GQA（全规模）** | GQA | GQA | GQA | GQA + MoE FFN |
| FFN | Dense FFN | Dense FFN | Dense FFN | Dense FFN | Dense FFN | Dense FFN | **MoE FFN（1 shared + N routed）** |
| 词表 | 32K SP-BPE | 32K SP-BPE | **128K BPE (tiktoken-style)** | 128K | 128K（vision tower 独立） | 128K | 128K + 视觉 patch tokens |
| 上下文 | 2K | 4K | 8K | **128K** | 128K | 128K | **1M（Maverick）/ 10M（Scout）** |

观察要点：
1. **架构核心不变**：从 LLaMA 1 到 LLaMA 4，Pre-RMSNorm + RoPE + SwiGLU 三件套**全程保留**，体现 Meta 认为这些"基础设施级选择"已被反复证明最优。
2. **GQA 三步走**：先在 LLaMA 2 70B 试水，证明 KV cache 收益巨大；LLaMA 3 全量普及；LLaMA 4 与 MoE 结合后仍是默认。
3. **MoE 转向**：直到 LLaMA 4 才采用 MoE，而非 LLaMA 3。原因：(a) Meta 在 405B Dense 上有完整工程信心，(b) MoE 在长上下文与多模态训练下的稳定性需要更多基础设施（路由 expert balance、推理引擎 KV/expert 调度等）；LLaMA 4 选择"晚到 MoE"是一次审慎的工程决策。
4. **长上下文跃升**：从 4K → 8K → 128K → 10M，每一代上下文增长约 8–80×，背后是 RoPE base scaling、长文档持续预训练、iRoPE / NoPE 等多种长上下文外推技术的组合。

### 5.2 训练规模与数据策略

| 模型 | 训练 tokens | 数据策略要点 |
| --- | --- | --- |
| LLaMA 1 65B | 1.4T | CommonCrawl + C4 + GitHub + Wikipedia + Books + ArXiv + StackExchange，公开数据 |
| LLaMA 2 70B | 2.0T | 上调高质量来源比例；PII 过滤；继续公开数据 |
| Code Llama | +500B 代码 | 在 LLaMA 2 之上 continued pretraining |
| LLaMA 3 70B | 15T+ | 比 LLaMA 2 多 4× 代码、5% 多语言；数据质量分类器过滤 |
| LLaMA 3.1 405B | 15.6T | 同 LLaMA 3 数据，但精细化退火（annealing）阶段大幅扩展数学 / 代码 / 推理 |
| LLaMA 3.2 1B/3B | — | 通过结构化剪枝 + 蒸馏自 8B/70B（不再独立大规模预训练） |
| LLaMA 4 Scout | ~40T | 文本 + 图像 + 视频帧多模态混合 |
| LLaMA 4 Maverick | ~22T + Behemoth 蒸馏 loss | 联合蒸馏预训练 |
| LLaMA 4 Behemoth | ~30T | 教师模型 |

数据质量策略的关键演进：
- **LLaMA 1**：粗粒度 dedup + 简单启发式；
- **LLaMA 2**：继续过滤 + 部分有害内容下采样；
- **LLaMA 3**：数据质量分类器 + 文本结构化分类（subject/topic/quality）+ 大规模 dedup（document/line/n-gram）+ 代码 / 多语言显式上调；
- **LLaMA 3.1**：在通用预训练之后引入 **退火阶段（annealing）**，专门用 0.5T 高质量推理 / 数学 / 代码数据再训练，类似"模型晚期素质教育"；
- **LLaMA 4**：原生多模态混合 + 来自 Behemoth 的蒸馏信号 + 丰富视频帧数据。

### 5.3 对齐范式（RLHF → DPO → 多轮迭代）

| 模型 | SFT | 偏好优化 | 备注 |
| --- | --- | --- | --- |
| LLaMA 2 Chat | ~28K 高质量 | **PPO + Rejection Sampling**（5 轮） | 双 RM（Helpfulness / Safety），首次工业级开源 RLHF |
| LLaMA 3 Instruct | ~10M 量级 | **SFT + Rejection Sampling + PPO + DPO** | 引入 DPO，与 PPO 并行 |
| LLaMA 3.1 Instruct | 多轮 | **6 轮 SFT → RS → DPO 迭代** | DPO 成为主力；405B 参与生成与拒绝采样 |
| LLaMA 4 | 重写的轻量 pipeline | **轻量 SFT → online RL → 轻量 DPO** | 谨慎避免"过拟合 to safety/format"；强调推理路径多样性 |

整体趋势：
- **从 PPO 主导 → DPO 主导**：DPO 因实现简单、不需要奖励模型在线打分，成为 2024-2025 主流；
- **从一锤子 → 多轮迭代**：每轮用最新模型自我标注 / 自我拒绝采样，构成 "self-play"-flavored alignment；
- **从重 SFT → 轻 SFT + 强 RL**：LLaMA 4 公开承认"过度 SFT 会损伤推理能力"，转向以 RL 为主的精炼阶段，这与 DeepSeek-R1、o1 系列的方向一致。

### 5.4 开源策略递进

| 版本 | 许可 | 关键变化 |
| --- | --- | --- |
| LLaMA 1 | 仅研究 / 学术 | 申请制；权重未公开下载（后泄漏） |
| LLaMA 2 | Llama 2 Community License | **首次开放商用**；月活 7 亿以上需另签 |
| Code Llama | Llama 2 Community License | 同 LLaMA 2 |
| LLaMA 3 / 3.1 / 3.2 / 3.3 | Llama 3.x Community License | 商用更便利；要求标注"Built with Llama" |
| LLaMA 4 | Llama 4 Community License | 仍带条件；月活 7 亿以上需申请 |

LLaMA 不是 Apache-2.0 / MIT 这样的"完全自由"许可，但其条件式开放足以让 99% 公司无障碍商用，是开源 LLM 商业化的事实门槛。

---

## 6. 开源生态影响

LLaMA 是 2023-2025 年开源 LLM 生态的"母体"，它催生的衍生家族至少可以分为五条主干：

### 6.1 指令微调家族（Alpaca → Vicuna → Guanaco → WizardLM …）

- **Alpaca**（Stanford, 2023-03）：基于 LLaMA-7B + 52K Self-Instruct（用 GPT-3.5 生成）数据指令微调；首次以**~600 美元**复刻接近 GPT-3.5 的指令遵循能力；引爆"小模型也能 follow 指令"叙事。
- **Vicuna**（LMSYS, 2023-03）：基于 LLaMA 13B + ShareGPT 用户对话指令微调；GPT-4 评估下达到 ChatGPT ~90% 能力；它的训练框架 *FastChat* 与评估方法 *MT-Bench* / *Arena* 至今仍是开源社区基础设施。
- **Guanaco / QLoRA**（Dettmers et al., 2023-05）：在 LLaMA 65B 上以 4-bit QLoRA 完成指令微调，**单卡 48GB** 即可训练 65B；直接催生 PEFT 在大模型时代的复兴。
- **WizardLM / WizardCoder / WizardMath**：用 *Evol-Instruct* 演化指令难度，把 LLaMA 衍生模型推到代码 / 数学专项 SOTA。
- **Tulu / Open-Hermes / Nous-Hermes / OpenChat / Zephyr** 等：每一波都依赖 LLaMA 家族基座。

### 6.2 多模态家族（LLaVA / MiniGPT-4 / mPLUG-Owl / VILA …）

- **LLaVA**（2023-04）：CLIP ViT + LLaMA / Vicuna，开创 visual instruction tuning；
- **MiniGPT-4 / Otter / VisualGLM** 等纷纷以 LLaMA 系作为语言塔；
- LLaMA 3.2 Vision 出现后，社区开始将其作为视觉对齐的**新一代基线**；
- LLaMA 4 native multimodal 上线后，原 cross-attention adapter 路线（LLaVA 风格）逐渐被 early fusion 取代。

### 6.3 PEFT 家族（LoRA / QLoRA / LLaMA-Adapter / LLaMA-Excitor）

- **LLaMA-Adapter / Adapter v2**：在 Transformer 高层 prepend 可学习 prompt token + zero-init gating，1.2M 参数完成指令 + 多模态适配；
- **QLoRA**：4-bit NF4 量化 + LoRA，是 PEFT 工程史上最具影响力的工作之一；
- **LLaMA-Excitor**：进一步在注意力上做"激励"调控，提升 MMLU 等 benchmark；
- 整个 HF *peft* 库的设计与 benchmark 选择，都深度围绕 LLaMA 家族展开。

### 6.4 国产化与多语言派生

- **Chinese-LLaMA**、**Chinese-Alpaca**、**OpenBuddy**、**TigerBot**、**Atom-7B** 等通过中文词表扩充 + 二次预训练，使 LLaMA 在中文场景可用；
- **LLaMA-Pro** / **Llama 2 Long** / **Llama-Pro Chinese** 探索深度持续学习方向。

### 6.5 模型蒸馏与"教师模型"地位

- LLaMA 3.1 405B 公开作为教师模型用于蒸馏 8B/70B；
- LLaMA 4 Behemoth (~2T) 显式作为蒸馏教师；
- 整个开源社区开始**默认**："最大尺寸 LLaMA 即业界免费的 GPT-4 替代教师"，把它用于：合成 SFT 数据、生成偏好对、做 judge model（LLM-as-a-judge）。

---

## 7. 关键创新点

将 LLaMA 系列贡献凝结为以下"标志性创新"：

1. **Pre-RMSNorm + RoPE + SwiGLU 三件套作为 Decoder-only Transformer 的事实标准**。LLaMA 1 不是这三者的发明者，但它把它们组合并验证至工业级，几乎所有 2023+ 开源 LLM 都直接复用。
2. **大模型 Pre-training 数据规模再校准**：LLaMA 1 公开提出"超出 Chinchilla optimal 训练 → 推理友好"，重塑了 Scaling Law 的工程实践；LLaMA 3 把 token 数推到 15T，进一步刷新基线。
3. **GQA 工业化**：Llama 2 70B 是 GQA 第一次成为开源旗舰的默认；LLaMA 3 全量普及，已成新常识。
4. **首个工业级开源 RLHF 全流程文档**：Llama 2 论文是 2023 年最被反复引用的对齐工程参考。
5. **Ghost Attention（GAtt）**：低成本解决多轮指令保持问题，已被多家社区项目复用。
6. **405B Dense 旗舰**：LLaMA 3.1 405B 是开源 Dense 模型的"天花板"；它证明 Dense 路线在工程上仍然可达，并通过它蒸馏小模型。
7. **6 轮 SFT-RS-DPO 迭代对齐**：LLaMA 3.1 推广这一 self-improving 后训练范式。
8. **128K → 10M 长上下文**：从 RoPE θ=5×10⁵ 到 iRoPE + NoPE 交错，每一步都为长文档 LLM 提供了可复用方案。
9. **首次开源 native multimodal MoE**：LLaMA 4 Scout / Maverick 是开源世界里第一个同时具备 MoE + 原生多模态 + 极长上下文的模型组合。
10. **教师模型 + 蒸馏闭环**：LLaMA 3.1 405B → 8B/70B；LLaMA 4 Behemoth → Maverick；为开源社区固化了"训一个大教师，蒸出一群可部署小模型"的工业范式。
11. **开源生态级影响**：Alpaca、Vicuna、QLoRA、LLaVA、Code Llama 等几乎所有 2023-2024 重要开源工作都直接以 LLaMA 为底座，构成最大的开源 LLM 社区根系。

---

## 8. 2025-2026 最新动态与展望

截至 2026 年 6 月（报告撰写时点），围绕 LLaMA 4 与后续路线的关键动态如下：

- **LLaMA 4 Scout / Maverick 已完成多轮迭代发布**：Hugging Face 与 llama.com 提供权重；社区围绕 Scout 10M 上下文展开了大量"single-prompt 处理整本书 / 整代码库"的工作。
- **LLaMA 4 Behemoth**：Meta 多次透露其"仍在训练并已超过 GPT-4-class 教师水平"，主要用于蒸馏，未来是否独立放出权重存在不确定性；2025-2026 间业界推测它将成为 Meta 内部产品（Meta AI 助手、Ray-Ban Meta、智能家居）的底座。
- **LLaMA 4 Reasoning**：Meta 在 2025 年路标中提及 Reasoning 子线（对标 OpenAI o1 / DeepSeek-R1），具体形态尚未稳定公开。
- **PEFT 与 Agent 工具链**：Llama Stack（含 Llama Agentic System、Code Shield、Prompt Guard）持续完善，是企业级 LLaMA 落地的官方推荐栈。
- **学术综述**：2025-10 出现的 *"Evolution of Meta's LLaMA Models and Parameter-Efficient Fine-Tuning of Large Language Models: A Survey"*（arXiv:2510.12178）系统梳理了 LLaMA 1–4 的演进与 PEFT 方法学，是当前最完整的 LLaMA 演进综述之一。
- **多模态生态**：随着 LLaMA 4 native multimodal 落地，社区的视觉指令数据（COCO、LLaVA-1.5/1.6 数据、MMMU、MMBench、ScienceQA）开始原生支持 LLaMA 4，而非再走 cross-attention adapter。
- **2026 路标传闻**：业内普遍预期 LLaMA 5 / LLaMA 4.5 会在以下方向继续推进：(a) 更细粒度专家（向 DeepSeek-V3 风格靠拢的可能）；(b) 显式 Reasoning RL pipeline；(c) 视频 / 音频原生模态；(d) Agentic 长任务的 long-horizon 评测与对齐。

总体判断：**LLaMA 系列已经从"开源 LLM 之一"演变为"开源大模型基础设施级公共品"**。它的下一个分水岭，可能会在 *"自研 reasoning RL 路线"* 与 *"细粒度 MoE 与原生 Agent 能力"* 两个方向上发生。

---

## 9. 参考文献

> 论文与官方资源；为简洁起见省略页码与版本号，按时间顺序组织。

### 9.1 LLaMA 主线论文与官方博客

1. Touvron, H. et al. *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971, 2023-02. <https://arxiv.org/abs/2302.13971>
2. Touvron, H. et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models.* arXiv:2307.09288, 2023-07. <https://arxiv.org/abs/2307.09288>
3. Rozière, B. et al. *Code Llama: Open Foundation Models for Code.* arXiv:2308.12950, 2023-08. <https://arxiv.org/abs/2308.12950>
4. Meta AI. *Introducing Meta Llama 3: The most capable openly available LLM to date.* 2024-04. <https://ai.meta.com/blog/meta-llama-3/>
5. Llama Team, Meta AI. *The Llama 3 Herd of Models.* arXiv:2407.21783, 2024-07. <https://arxiv.org/abs/2407.21783>
6. Meta AI. *Introducing Llama 3.1: Our most capable models to date.* 2024-07. <https://ai.meta.com/blog/meta-llama-3-1/>
7. Meta AI. *Llama 3.2: Revolutionizing edge AI and vision with open, customizable models.* 2024-09. <https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/>
8. Meta AI. *Llama 3.3 70B Model Card.* 2024-12. <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>
9. Meta AI. *The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation.* 2025-04. <https://ai.meta.com/blog/llama-4-multimodal-intelligence/>
10. Meta AI. *Llama 4 official site.* <https://www.llama.com/models/llama-4/>

### 9.2 关键基础技术

11. Su, J. et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864, 2021.
12. Shazeer, N. *GLU Variants Improve Transformer.* arXiv:2002.05202, 2020. （SwiGLU）
13. Zhang, B., Sennrich, R. *Root Mean Square Layer Normalization.* NeurIPS, 2019. （RMSNorm）
14. Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* arXiv:2305.13245, 2023.
15. Hu, E. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021.
16. Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314, 2023.
17. Rafailov, R. et al. *Direct Preference Optimization.* arXiv:2305.18290, 2023.（DPO）

### 9.3 LLaMA 衍生与生态

18. Taori, R. et al. *Stanford Alpaca: An Instruction-following LLaMA model.* 2023.
19. Chiang, W.-L. et al. *Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality.* LMSYS Org, 2023-03. <https://lmsys.org/blog/2023-03-30-vicuna/>
20. Liu, H. et al. *Visual Instruction Tuning (LLaVA).* NeurIPS, 2023.
21. Zhang, R. et al. *LLaMA-Adapter / LLaMA-Adapter V2.* arXiv:2303.16199, 2023.
22. Xu, C. et al. *WizardLM: Empowering Large Language Models to Follow Complex Instructions.* arXiv:2304.12244, 2023.

### 9.4 综述与第三方分析

23. Abdulla, A. A. et al. *Evolution of Meta's LLaMA Models and Parameter-Efficient Fine-Tuning of Large Language Models: A Survey.* arXiv:2510.12178, 2025-10. <https://arxiv.org/abs/2510.12178>
24. Wolfe, C. *Llama 4: The Challenges of Creating a Frontier-Level LLM.* Substack, 2025. <https://cameronrwolfe.substack.com/p/llama-4>
25. Wolfe, C. *Beyond LLaMA: The Power of Open LLMs.* Substack, 2023.
26. Hugging Face. *Welcome Llama 4 Maverick & Scout on Hugging Face.* 2025-04. <https://huggingface.co/blog/llama4-release>
27. Hugging Face. *Llama 3.1 - 405B, 70B & 8B with multilinguality and long context.* 2024-07. <https://huggingface.co/blog/llama31>
28. Devopedia. *Llama (LLM).* <https://devopedia.org/llama-llm>
29. Wikipedia. *Llama (language model).* <https://en.wikipedia.org/wiki/Llama_(language_model)>
30. Linsight. *Llama 3.1 预训练 / post-training 要点一览.* <https://saicat.github.io/7d7294cb.html>

### 9.5 对比模型（用于跨厂商参照）

31. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024-12.
32. DeepSeek-AI. *DeepSeek-R1.* 2025.
33. Mistral AI. *Mixtral of Experts.* arXiv:2401.04088, 2024.
34. Qwen Team. *Qwen2 / Qwen2.5 / Qwen3 Technical Reports.* 2024-2025.

---

> **撰写说明**：本报告基于 arXiv 公开论文、Meta AI 官方博客、Hugging Face 模型卡、第三方综述与社区分析综合整理；未访问 Meta 内部材料。引用如有版本差异（特别是 LLaMA 4 仍在演化）以最新官方发布为准。
