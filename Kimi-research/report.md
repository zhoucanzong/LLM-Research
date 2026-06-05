# Kimi 系列模型深度调研报告

> 调研对象：Moonshot AI（月之暗面）的 Kimi 模型家族
> 时间跨度：2023 年 10 月 ~ 2026 年 6 月
> 资料来源：arXiv 论文、Moonshot AI 官方技术博客、HuggingFace 模型卡、GitHub 仓库
> 撰写日期：2026 年 6 月

---

## 1. 概述与时间线

Moonshot AI（月之暗面，Beijing Yuezhi Technology Co., Ltd.）由清华大学校友、原 Recurrent.ai 创始人**杨植麟（Zhilin Yang）**于 2023 年 3 月在北京创立，2023 年 10 月发布首款 C 端产品 **Kimi 智能助手**，成为中国最早押注「超长上下文（Long-Context）」路线的大模型公司之一。其技术演进沿着三条主线展开并彼此交织：

1. **长上下文主线**：Moonshot-v1（200K）→ 2M 字符内测 → **MoBA**（块级稀疏注意力）→ **Kimi Linear / KDA**（线性注意力混合架构），系统性回答「如何让 Transformer 真正吃下百万 token」。
2. **推理（Reasoning）主线**：**Kimi k1**（2024.11，多模态思考预览）→ **Kimi k1.5**（2025.01，对标 OpenAI o1）→ **Kimi K2 Thinking**（2025.11，对标 GPT-5、Claude Sonnet 4.5），形成「以长上下文 RL + Long2Short」为核心的独特路线。
3. **基础模型与多模态主线**：**Moonlight 16B-A3B**（2025.02，首个用 Muon 优化器训练的 MoE）→ **Kimi-VL / Kimi-VL-Thinking**（2025.04）→ **Kimi-Audio**（2025.04）→ **Kimi K2 1T-A32B**（2025.07，开源「Open Agentic Intelligence」）→ **Kimi K2.5**（2026.01，原生多模态 Agent）→ **Kimi K2.6**（2026.04，开源编码/多 Agent 集群旗舰）→ **K2 系列退役**（2026.05）。

### 1.1 关键时间线

| 时间 | 事件 | 说明 |
| --- | --- | --- |
| 2023-03 | Moonshot AI 成立 | 杨植麟、周昕宇（Xinyu Zhou）、吴育昕（Yuxin Wu）创立 |
| 2023-10-09 | Kimi Chat 上线 | 首发即支持 **20 万汉字**（约 200K characters）无损上下文 |
| 2024-03 | 200 万汉字内测 | 全球首个开放 **2M Chinese characters** 长上下文的 C 端产品 |
| 2024-08 | Moonshot-v1 API（8K/32K/128K） | 三档上下文 API 商业化 |
| 2024-11 | **Kimi k1 视觉思考模型**预览 | 中国首个开放多模态长思考模型 |
| 2025-01-22 | **Kimi k1.5** 技术报告（arXiv:2501.12599） | 与 DeepSeek-R1 同期，o1 级推理性能 |
| 2025-02-21 | **Moonlight / Muon 论文**（arXiv:2502.16982） | 首次将 Muon 扩展到 16B MoE 规模 |
| 2025-02-18 | **MoBA 论文**（arXiv:2502.13189） | 块级稀疏注意力，已用于 Kimi 线上长上下文 |
| 2025-04-10 | **Kimi-VL Technical Report**（arXiv:2504.07491） | 2.8B 激活的 MoE VLM，附 Kimi-VL-Thinking |
| 2025-04-25 | **Kimi-Audio**（arXiv:2504.18425） | 7B 音频基础模型 |
| 2025-07-22 | **Kimi K2** 开源（arXiv:2507.20534） | 1T 总 / 32B 激活 MoE，**MuonClip** 优化器 |
| 2025-10-30 | **Kimi Linear** 论文（arXiv:2510.26692） | KDA + MLA 3:1 混合，1M 上下文 6× 解码加速 |
| 2025-11-06 | **Kimi K2 Thinking** 发布 | 1T MoE + 256K 上下文 + 原生 INT4，HLE 44.9%（带工具） |
| 2026-01-27 | **Kimi K2.5** | 在 K2 基础上继续预训练 ~15T 视觉-文本 token，原生多模态 Agent，Agent Swarm 100 子代理 |
| 2026-04-13 | **Kimi Code K2.6** | 编码代理工具发布 |
| 2026-04-20 | **Kimi K2.6** 开源 | 1T 参数 / 32B 激活 MoE，256K 上下文，SWE-Bench Verified 80.2%（开源模型第一），Modified MIT License，API $0.95/$4.00 per 1M tokens |
| 2026-05-25 | **K2 系列退役** | K2 系列正式停止支持，仅 K2.6 继续维护 |
| 2026-06 | **Claw Groups / 视频输入** | 异构代理集群研究预览；支持 mp4, mov, webm, avi 视频输入 |

> 注：本报告的「推理模型」与「基座模型」在 2025 年 11 月之后开始合流——K2 Thinking 起，Moonshot 选择**单一旗舰多形态**（base / instruct / thinking）的发布范式，与 DeepSeek 模式趋同。

---

## 2. 基座模型演进：以长上下文为主线

### 2.1 第一代 Moonshot-v1（2023-2024）：注意力工程与基础设施驱动的 200K → 2M

Kimi 在 2023 年 10 月以「20 万汉字无损上下文」首次进入大众视野时，主流闭源模型仍停留在 GPT-4 的 8K~32K 区间（GPT-4 Turbo 128K 直到 2023 年 11 月才发布）。此时 Moonshot-v1 并未发布详细技术报告，从后续披露与官方开放平台信息可推断其技术路径：

- **基础架构**：Transformer Decoder + RoPE（Rotary Position Embedding），由 Moonshot 联合创始人苏剑林（Jianlin Su，RoPE 原作者）主导设计。
- **长上下文实现**：分阶段课程式训练（curriculum）逐步将上下文从 8K → 32K → 128K → 200K 拓展，配合 RoPE base frequency 调整与 YaRN 系列外推方法。
- **基础设施**：自研分布式训练栈（Sequence/Context Parallelism + ZeRO + 重计算）以承载 200K+ 序列；KV-Cache 分页与持久化存储用于推理期长对话上下文复用。
- **2024-03 的 2M 中文字符内测**：对外宣称「无损」（lossless），实际依赖大段文档的离线预处理与上下文压缩 / 检索增强；这是产品形态而非纯模型能力升级，但确立了 Kimi 在 C 端「长文本助手」的品牌定位。

> **结论**：第一代 Kimi 的「长上下文领先」更多来自系统工程层面的胜利——通过基础设施换取上下文长度，而非根本性的注意力革新。这一阶段为后续真正的算法创新（MoBA、KDA）积累了工程基础和训练数据。

### 2.2 MoBA：从 Dense Attention 到块级稀疏（2025-02）

随着 1M token 级别上下文需求出现，标准 Softmax Attention 的 O(n²) 复杂度成为不可逾越的成本墙。Moonshot 给出的第一版算法答卷是 **MoBA（Mixture of Block Attention，arXiv:2502.13189）**：

- **核心思想**：将 MoE 路由思想搬到注意力层。把序列切成固定大小的 block，由可学习的 router 为每个 query 选择 top-k 个最相关 block 进行注意力计算，其余 block 完全跳过。
- **「Less Structure」原则**：不像 sliding-window、StreamingLLM 那样硬编码注意力 pattern，让模型自行决定关注哪些块。
- **可切换性**：MoBA 支持 full-attention 与 sparse-attention 之间无缝切换——预训练用 full，长上下文阶段切换到 MoBA；这一性质是 DeepSeek NSA、Native Sparse Attention 等同期工作所不具备的。
- **工程价值**：论文明确指出 **MoBA 已部署于 Kimi 线上服务**，承担长上下文请求的实际推理流量。这意味着 Kimi 的 1M 上下文不是「能跑」而是「跑得起」。

### 2.3 Kimi Linear / KDA：线性注意力首次「全面战胜」全注意力（2025-10）

MoBA 解决了「能不能算」的问题，但仍是 Softmax 衍生品；KV Cache 仍随上下文线性增长，1M token 推理时显存压力巨大。**Kimi Linear（arXiv:2510.26692）**给出更激进的答案：

- **Kimi Delta Attention（KDA）**：在 Gated DeltaNet 之上引入更精细的**逐通道（fine-grained）门控**与改进的 delta-rule 状态更新规则，本质上是带有「遗忘门」的线性 RNN，状态尺寸恒定（不随序列长度增长）。
- **3:1 混合架构**：每 4 层中 3 层 KDA + 1 层 **MLA（Multi-head Latent Attention，DeepSeek-V3 风格的全注意力）**，并对所有 MLA 层采用 **NoPE（No Positional Encoding）**——把位置编码「外包」给 KDA 的递归状态，全局层只负责语义聚合。
- **指标**：在 1.4T token 公平对照预训练下，48B-A3B 的 Kimi-Linear-Base 在 MMLU-Pro 拿 51.0、RULER（128K）拿 84.3，**首次出现线性注意力混合架构在通用 + 长上下文 + RL 三条赛道全面优于全注意力**。
- **效率**：相比纯 MLA，**KV Cache 减少 75%、1M 上下文 TPOT（Time-Per-Output-Token）加速 6×**。
- **意义**：这是 Moonshot 自 RoPE 之后在注意力机制上的第二次范式级别贡献，也是其后续 K2.x 系列模型的重要架构候选。

| 阶段 | 代表 | 注意力类型 | 上下文 | KV 增长 | 实际负载 |
| --- | --- | --- | --- | --- | --- |
| 2023-2024 | Moonshot-v1 | Dense Softmax + RoPE | 200K (后扩 2M) | O(n) | 通过工程承载 |
| 2025-Q1 | MoBA | 块级稀疏 + 路由 | ≥1M | O(n) 但常数小 | 已上线 Kimi |
| 2025-Q4 | Kimi Linear | KDA(线性)+MLA(NoPE) 3:1 | 1M | KDA O(1)，MLA O(n)/4 | 候选下一代基座 |

### 2.4 MoE 探索：从 Moonlight 到 K2 的稀疏化路径

Moonshot 在 2025 年完整跑通了 MoE：

- **Moonlight 16B-A3B（2025-02）**：Moonshot 首个 MoE 基座，3B 激活 / 16B 总参数，5.7T token 训练；架构沿用 DeepSeek-V3 family（细粒度专家 + 共享专家 + MLA）。**首次大规模应用 Muon 优化器**，相比 AdamW 在相同算力下达成 ~2× token 效率（详见 §6.1）。
- **Kimi K2 1T-A32B（2025-07）**：61 层、384 路由专家 + 1 共享专家 + 64 注意力头 + MLA。沿 DeepSeek-V3 蓝图但做出三个关键差异化选择：
  1. **稀疏度提升至 48**（DeepSeek-V3 为 32）：8/384 激活，论文给出「sparsity scaling law」证明在固定 FLOPs 下稀疏度越高 loss 越低；
  2. **注意力头减半**（64 vs 128）：以略微 loss 退化（0.5–1.2%）换取长上下文推理速度；
  3. **取消 expert grouping**：换更小 EP=16 配合 1F1B，避免 DualPipe 的额外内存。
- **Kimi-VL 16B-A3B（2025-04）** 复用 Moonlight 架构作为语言塔；**Kimi-Linear 48B-A3B（2025-10）**则是更新一代稀疏度更高的 MoE。

| 模型 | 总参数 | 激活参数 | 路由专家数 | 共享专家 | 稀疏度 | Attention | 优化器 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Moonlight | 16B | 3B (2.8B) | — | — | 中等 | MLA | Muon |
| Kimi-VL | 16B | 2.8B + 0.4B ViT | — | — | 中等 | MLA | Muon (enhanced) |
| Kimi K2 | 1.04T | 32.6B | 384 | 1 | 48 | MLA, 64 heads | **MuonClip** |
| Kimi-Linear | 48B | 3B | — | — | 16 | KDA + MLA(NoPE) 3:1 | Muon |
| Kimi K2.5 | 1T 级 | 32B 激活 | 沿用 K2 | 1 | 48 | MLA | MuonClip + 续训 |
| **Kimi K2.6** | **1T** | **32B** | **沿用 K2** | **1** | **48** | **MLA** | **MuonClip + 续训** |

---

## 3. 推理模型：Kimi k1 / k1.5 与 K2 Thinking

### 3.1 Kimi k1（2024-11）：中国首个多模态长思考模型预览

Kimi k1 是 OpenAI o1 公布（2024-09）后业界最早的对标产品之一，与 DeepSeek-R1-Lite-Preview 同期。官方未发独立论文，从 k1.5 报告与博客可拼出技术骨架：

- 多模态原生（图像 + 文本同栈），AIME / MathVista 等多任务超越 GPT-4o；
- 长 CoT 由 SFT「激活」+ 简单 RL 强化得到；
- 这一版的最大价值是建立了**长上下文 + RL** 的研究 hypothesis，为 k1.5 铺路。

### 3.2 Kimi k1.5（2025-01-22）：长上下文 RL 的成功实践

Kimi k1.5（arXiv:2501.12599）与 DeepSeek-R1 同日（2025-01-20）开放，是 2025 年开源/半开源推理模型最重要的两份报告之一。其核心论点是：**长上下文是 RL 的另一条 scaling axis，不依赖 MCTS / Value Function / PRM 也能逼近 o1**。

#### 3.2.1 关键技术

| 技术 | 描述 | 与 R1 / o1 对比 |
| --- | --- | --- |
| **Long Context Scaling** | RL 上下文窗口扩到 **128K**，长 CoT 自然涌现规划、反思、回溯 | R1 同样长 CoT，但更强调 GRPO；o1 未公开 |
| **Partial Rollout** | 复用前次 trajectory 的大段，仅采样新 token，避免完整重算 | 创新点；R1 / o1 未公开类似机制 |
| **Online Mirror Descent (变体)** | 不用 PPO 的 clip ratio，用相对熵 KL 正则的策略镜像下降 | 与 R1 的 GRPO 不同路线，更接近经典 RLHF |
| **Length Penalty** | 显式惩罚冗余长度，缓解 overthinking | R1 让模型自发收敛；k1.5 显式约束 |
| **Curriculum + Prioritized Sampling** | 用难度标签与每实例成功率聚焦「中等难度」样本 | 类似 OpenAI o1 的样本难度策划 |
| **拒绝复杂结构** | 不用 MCTS / Value Function / Process Reward Model | 与 R1 一致：只用最终结果奖励 |

奖励目标：

$$\max_{\theta}\mathbb{E}_{(x,y^*)\sim\mathcal{D}}\Big[\mathbb{E}_{(y,z)\sim\pi_{\theta}}[r(x,y,y^*)] - \tau\,\mathrm{KL}(\pi_{\theta}(x)\,\|\,\pi_{\theta_i}(x))\Big]$$

其中 $r\in\{0,1\}$ 仅依据最终答案 $y$ 与 ground truth $y^*$ 是否一致，$\pi_{\theta_i}$ 是上一轮迭代策略，本轮目标是相对它做 KL 受限的 mirror descent。

#### 3.2.2 Long2Short：把长 CoT 蒸馏回短 CoT

k1.5 提出 **long2short** 范式来兼顾推理质量与推理成本：

- **Model Merging**：直接对 long-CoT 和 short-CoT 模型做权重平均；
- **Shortest Rejection Sampling**：多次采样取最短正确答案；
- **DPO**：以「长正确 vs 短正确」对作为 chosen，「冗长错误」作为 rejected；
- **Long2Short RL**：把 short-CoT 模型在长 CoT 数据上做带显式长度奖励的二阶段 RL。

最终 short-CoT 版本：AIME 60.8 / MATH500 94.6 / LiveCodeBench 47.3，超越 GPT-4o、Claude 3.5 Sonnet **最高 +550%**。

#### 3.2.3 多模态原生

k1.5 不是「文本模型 + 视觉适配器」，而是**文本与视觉数据联合 RL 训练**，MathVista 74.9 与 o1 持平。这一选择直接孕育了 §4 的 Kimi-VL-Thinking。

### 3.3 Kimi K2 Thinking（2025-11-06）：开源对标 GPT-5

K2 Thinking 是在 Kimi K2（base / instruct）之上做进一步的「思考态 RL」得到，定位为 Moonshot 第一代「真正可作 Agent 大脑」的开源模型：

- **架构**：1T MoE / 32B 激活，**256K 上下文窗**（K2 base 是 128K，K2 Thinking 进一步扩展并配合 YaRN）；
- **原生 INT4 推理**：在 post-training 阶段用 **Quantization-Aware Training（QAT）**对 MoE 部分做 INT4 weight-only 量化，所有公开 benchmark 均在 INT4 精度下报告，推理速度~2×；
- **Interleaved Thinking + 多步工具调用**：可在多达 **200~300 次顺序工具调用**之间穿插 thinking token，是首批具备这种能力的开源模型（同期为 MiniMax M2 2025-11-03）；
- **关键 benchmark**（INT4，官方）：
  - Humanity's Last Exam（HLE，with tools）：**44.9%**
  - BrowseComp：**60.2%**
  - SWE-Bench Verified（base K2 已达 65.8%）等。

### 3.4 与 DeepSeek-R1 的异同

| 维度 | Kimi k1.5 / K2 Thinking | DeepSeek-R1 / R1-Zero |
| --- | --- | --- |
| RL 算法 | Online Mirror Descent + KL 正则 | GRPO（Group Relative Policy Optimization） |
| 长上下文 | **核心维度**，RL 阶段直接扩到 128K-256K | 长 CoT 自然涌现，未把 context 当 scaling axis |
| 价值函数 / PRM | 不使用 | 不使用 |
| Long2Short | **显式系统化方法**（merge / DPO / 长度奖励 RL） | 主要靠 R1 → 蒸馏小模型 |
| 多模态原生 | **是**（文本+视觉联合 RL） | R1 仅文本 |
| 架构 | k1.5 多模态稠密 → K2 Thinking 1T MoE+MLA | R1 用 V3 671B MoE+MLA |
| 优化器 | Muon → MuonClip | AdamW |
| 量化 | K2 Thinking 原生 INT4 QAT | 后置量化 |

> **结论**：k1.5 与 R1 是 2025 年 1 月开源推理模型的「双子星」。R1 重在 RL 算法（GRPO + R1-Zero 纯 RL 路线）；Kimi 重在**长上下文 RL + 多模态 + Long2Short 工程闭环**。两条技术线后续相互渗透——K2 Thinking 已经在算法上吸收了部分 GRPO 思想，DeepSeek 后续模型也开始重视长上下文与稀疏注意力。

---

## 4. 多模态模型：Kimi-VL / Kimi-VL-Thinking / Kimi-Audio

### 4.1 Kimi-VL（2025-04）：高效 MoE 视觉语言模型

Kimi-VL（arXiv:2504.07491）是 Moonshot 第一份正式的 VLM 技术报告，主打「**轻而强**」——2.8B+0.4B 激活参数对标 GPT-4o-mini / Qwen2.5-VL-7B / Gemma-3-12B-IT。

#### 4.1.1 架构三件套

```
图像  →  MoonViT (400M, native resolution)  →  Pixel-Shuffle 2×2  →  2-layer MLP  →  Moonlight MoE (16B/2.8B-A3B)
```

- **MoonViT（原生分辨率视觉编码器）**：吸收 NaViT 的 packing 思想——将不同分辨率的图像切 patch、展平、序列拼接，因此可与 LLM 共享 FlashAttention 这类变长序列注意力 kernel。**不需要 LLaVA-OneVision 风格的 sub-image 切块/拼接**。
- **MLP Projector**：先 pixel-shuffle 把 2×2 空间维度折到通道维（特征数 ×4，序列长度 ÷4），再两层 MLP 映射到 LLM hidden。
- **Moonlight MoE 解码器**：16B/2.8B-A3B，结构同 DeepSeek-V3 family；从 Moonlight 中间 checkpoint（5.2T token、8K 上下文）继续训。

> **MoE 应用于 VLM 是 Kimi-VL 的关键差异点**：相比同期主流 dense VLM（Qwen2.5-VL-7B、Gemma-3-12B），它在 24 个 benchmark 中 19 项超越 Qwen2.5-VL-7B（后者激活 2.59× 参数），证明 MoE 路线在 VLM 上同样有效。

#### 4.1.2 训练策略：四阶段共 4.4T token

| 阶段 | 数据 | 序列长度 | 可训练 |
| --- | --- | --- | --- |
| ViT 训练 | Alt-text + 合成 caption + grounding + OCR (2.1T) | 8192 | 仅 ViT |
| 联合预训练 | + 文本/知识/交错图文/视频/Agent (1.4T) | 8192 | ViT + LLM |
| 联合 cooldown | + 高质量文本与多模态、学术语料 (0.6T) | 8192 | ViT + LLM |
| **联合长上下文激活** | 长文本/长视频/长文档 + 25% 长 + 75% 短 (0.3T) | **32K → 128K** | ViT + LLM |

最后一阶段把 RoPE base 从 50,000 → 800,000，分两段 4× 扩展。NIAH（Needle-in-a-Haystack）测试在 128K 文本/视频上检索准确率仍 ≥87%/91.7%。

#### 4.1.3 Kimi-VL-Thinking：原生多模态思考

在 Kimi-VL 之上：
1. **Long-CoT SFT**：用长链思考监督微调，激活长思考能力；
2. **RL（同 k1.5）**：online mirror descent + KL 正则 + 长度奖励 + 课程/优先级采样；最终在 MMMU 61.7、MathVision 36.8、MathVista 71.3 全面对标更大模型。

这是与「外接 reasoning 模块」截然不同的范式：**视觉与思考过程在同一个模型内端到端 RL 训练**，与 OpenAI o1（视觉版）思路一致，但开源、且只用 2.8B 激活参数。

### 4.2 Kimi-Audio（2025-04）

Kimi-Audio（arXiv:2504.18425）是 Moonshot 的音频基础模型，7B 参数，基于 Qwen2.5-7B 并融合 Whisper 编码思路，统一支持 ASR、AQA（音频问答）、TTS、音频对话等。它与 Kimi-VL 同月发布，构成 Moonshot 多模态矩阵的语音侧。

### 4.3 Kimi K2.5 / K2.6：原生多模态 Agent

- **Kimi K2.5（2026-01-27）**：在 K2（纯文本）基础上做约 **15T 视觉 + 文本混合 token 的 continual pretraining**，定义「Visual Agentic Intelligence」——同一个 1T MoE 同时承担文本推理、视觉理解、多步 Agent 执行；强调视觉编码（screenshots、UI、图表）下的 Agent 行为。支持 **Agent Swarm 100 子代理** 协同。
- **Kimi K2.6（2026-04-20）**：开源发布，1T 参数 / 32B 激活 MoE，**256K 上下文**，采用 **Modified MIT License**。关键指标：
  - **SWE-Bench Verified 80.2%**——开源模型第一；
  - **SWE-Bench Pro 58.6%**——与 GPT-5.5 持平；
  - **Agent Swarm 300 子代理 / 4000 步**——面向超长程软件工程任务；
  - 官方披露可**连续编码 13 小时**；
  - **API 定价**：$0.95 / $4.00 per 1M tokens（输入/输出）。
- **Kimi Code K2.6（2026-04-13）**：编码代理工具，面向 IDE 集成与自动化软件工程。
- **视频输入支持**：新增 mp4、mov、webm、avi 格式视频输入能力。
- **Claw Groups（研究预览）**：异构代理集群（Heterogeneous Agent Cluster），支持不同类型 Agent 协同完成复杂任务。

至此，Moonshot 的多模态路线呈现「**小而强（VL/Audio 单独发版）→ 大而全（K2.5 单一旗舰原生多模态）→ 开源旗舰（K2.6 开源 + 编码工具）**」的演进。

---

## 5. 跨代技术演进分析

### 5.1 三条主线的合流

```
长上下文        2023.10 200K  →  2024.03 2M  →  2025.02 MoBA  →  2025.10 KDA
                                   │                │              │
                                   └──── 共享 Infra ─┴──────────────┤
                                                                    │
推理            2024.11 k1  →  2025.01 k1.5  →  2025.11 K2 Thinking ┤
                              (RL+128K)         (1T MoE+256K+INT4) ─┤
                                                                    │
基座 / MoE      2025.02 Moonlight  →  2025.07 K2  →  2026.01 K2.5 ──┘
                (16B-A3B, Muon)     (1T-A32B,        (原生多模态)
                                     MuonClip)
```

至 K2 Thinking / K2.5 阶段，三条主线收敛于同一个 1T MoE backbone：长上下文（256K）、推理（thinking）、多模态（K2.5）皆基于同一架构。

### 5.2 注意力机制演进

| 时代 | 模型 | 注意力 | 位置编码 | KV 复杂度 |
| --- | --- | --- | --- | --- |
| 2023-2024 | Moonshot-v1 | 多头全注意力 | RoPE | O(n) |
| 2024-2025 | Moonlight / Kimi-VL / K2 | **MLA** (DeepSeek-V3 风格) | RoPE | O(n)，但 latent 压缩 |
| 2025-Q1 | Kimi 线上长上下文 | **MoBA** | RoPE | O(n)，常数小 |
| 2025-Q4 | Kimi-Linear | **KDA(3) + MLA-NoPE(1)** 混合 | KDA 内禀位置 + MLA NoPE | KDA O(1)，混合显著低于 MLA |

**MLA、MoBA、KDA 同时由 Moonshot 团队（部分作者重叠 Jianlin Su、Enzhe Lu、Jingyuan Liu 等）推动**，构成中国大模型注意力机制创新的完整谱系。

### 5.3 训练优化器：Muon → MuonClip

Moonshot 是把 Muon 优化器从「100M 玩具实验」推到「1T 工业训练」的关键团队，详见 §6.1-§6.2。

### 5.4 后训练范式

| 代际 | SFT | RL | 偏好 / 自评 | 工具 / Agent |
| --- | --- | --- | --- | --- |
| Moonshot-v1 | 标准指令 SFT | RLHF (PPO 类) | RM | 简单工具 |
| k1.5 / Kimi-VL-Thinking | 长 CoT SFT | **Online Mirror Descent + KL** + Length Penalty | 结果奖励 (0/1) | 多模态 |
| K2 (Instruct) | 大规模 Agentic SFT（合成 20K+ 工具，多 Agent 轨迹） | **Verifiable Rewards Gym + Self-Critic Rubric** | 自评成对比较 | **20K+ 合成工具**，K8s sandbox 1 万并发 |
| K2 Thinking | + Long-CoT thinking SFT | + Interleaved-thinking RL | + Faithfulness Judge | **200-300 步顺序工具调用** |
| K2.5 / K2.6 | + 视觉 Agent SFT | + 多 Agent 协同 RL | — | **300 子 Agent 集群、13h 连续编码、SWE-Bench 80.2%** |

后训练每代迭代的「门」是 Agent / 工具调用复杂度的指数级增长。

---

## 6. 关键创新点详述

### 6.1 Muon 优化器：从 100M 玩具到 16B MoE（Moonlight 论文）

**Muon**（Jordan 等，2024）原本是基于 Newton-Schulz 矩阵正交化的小规模优化器。Moonlight 论文（arXiv:2502.16982）做了三项关键工作把它推上主流：

1. **加权重衰减（Weight Decay）**：原始 Muon 不带 weight decay，scaling 到 800M+ 时观察到权重 RMS 持续增长甚至超出 bf16 高精度区间；引入 AdamW 风格 $\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t(\mathbf{O}_t + \lambda\mathbf{W}_{t-1})$ 后既稳定又能在 over-train regime 优于 AdamW。
2. **一致性 Update RMS**：证明对 $[A,B]$ 形状参数 Muon 理论 update RMS 为 $\sqrt{1/\max(A,B)}$，提出乘以 $\sqrt{\max(A,B)}$ 修正，并整体缩放至 0.2 与 AdamW 匹配——使 Muon 可**直接复用 AdamW 调好的学习率与 weight decay**，不需重新调参。
3. **分布式 Muon**：基于 ZeRO-1 分片，引入 DP-Gather 收集完整梯度矩阵 → Newton-Schulz 迭代 → 丢弃非本地分片；通信开销仅为 AdamW 的 1×~1.25×，实际训练里几乎无显著 overhead；BF16 通信进一步省 50%。

Scaling-law 实验显示 **Muon 仅需 ~52% AdamW 的 FLOPs 即可达到等效性能**。Moonlight 16B-A3B 用 Muon 训 5.7T token，在 MMLU、HumanEval、GSM8K 等推到当时同档 Pareto 前沿。

### 6.2 MuonClip：QK-Clip 解决万亿规模注意力 Logit 爆炸

K2 论文（arXiv:2507.20534）发现：在 9B-A53B 的 MoE 中训练时 Muon 仍会让 attention logits 爆到 1000+，而 AdamW 不会；这说明 Muon 的「token efficiency 优势」与「logit 不稳定」相伴而生。

**MuonClip = Muon + QK-Clip**：

1. 每步前向算每头最大 logit $S_{\max}^h = \frac{1}{\sqrt{d}}\max_{X\in B}\max_{i,j}\mathbf{Q}_i^h\mathbf{K}_j^{h\top}$；
2. 若 $S_{\max}^h > \tau$（K2 用 $\tau=100$），按 $\gamma_h = \tau/S_{\max}^h$ 对 $\mathbf{W}_q^h, \mathbf{W}_k^h$ 做权重重缩放（**逐头**而非全局，干预最小化）；
3. 对 MLA 的特殊情况：
   - 头特定的 $q^C, k^C$：各乘 $\sqrt{\gamma_h}$
   - 头特定的旋转 $q^R$：乘 $\gamma_h$
   - **共享 rotary $k^R$ 不动**（避免跨头串扰）

K2 在此基础上 15.5T token 全程**零 loss spike**——这是 1T 级开源 MoE 训练史上最干净的 loss 曲线之一，也是 MuonClip 最重要的产业级证明。

### 6.3 Kimi K2 的 Agentic 数据合成与 RL

K2 的非思考态 SoTA（SWE-Bench Verified 65.8、Tau2-Bench 66.1、ACEBench-EN 76.5）主要来自三方面：

#### 6.3.1 大规模 Agentic 数据合成

- **工具规格生成**：3000+ 真实 MCP 工具（GitHub） + 通过 hierarchical domain evolution 合成 **20,000+ 合成工具**；t-SNE 显示二者在工具空间互补覆盖；
- **Agent 多样化**：上千 system prompt × 工具组合 = 数千 agent；
- **Rubric-based 任务生成**：每个任务带显式 rubric（成功标准、期望工具调用 pattern、检查点）；
- **多轮轨迹生成**：LLM 模拟用户 + LLM 模拟工具执行环境（带状态、有控制随机性的成功/部分失败/边界）；
- **质量过滤**：LLM-as-judge 按 rubric 筛选高质量轨迹用于 SFT。

#### 6.3.2 通用 RL：Verifiable Rewards Gym + Self-Critic Rubric

- **可验证奖励**（数学/STEM/编程）：用 SFT 模型 pass@k 选「中等难度」样本，避免太易 / 太难产生稀疏信号；
- **复杂指令遵循**：双路径——代码执行验证 + LLM-as-judge + hack-check 防伪；
- **Faithfulness Judge**：句子级事实核查 reward model；
- **Self-Critic Rubric Reward**：对创意写作 / 开放问答这类没有 ground truth 的任务，让模型自己做成对比较打分；
- **统一 Gym-like 框架**：一套 RL 框架同时容纳数学、代码、SWE、指令遵循、创意写作。

#### 6.3.3 训练 Recipe（K2 Pre-train）

- 4096 token 上下文 + WSD（warm-stable-decay）学习率，500 步 warmup → 10T token 恒定 2e-4 → 5.5T token cosine decay 到 2e-5；
- 长上下文激活：4K → 32K → **YaRN 扩到 128K**；
- 16-way PP + 16-way EP + ZeRO-1 DP + 选择性重计算 + FP8 存储非敏感激活 + Activation CPU Offload；H800 集群。

### 6.4 Kimi K2 Thinking 的原生 INT4 QAT

K2 Thinking 在后训练阶段对 MoE 部分做 **INT4 weight-only QAT**，所有公开 benchmark 直接报告 INT4 数字。这一点的工程意义大于学术意义：

- **公平性**：用户看到的就是部署版本性能；
- **训练 RL 长序列友好**：INT4 让 thinking RL 在长 trace 上的显存压力减半；
- **2× 推理加速** + 无明显精度损失，进一步扩大开源模型在「免费 + 廉价部署」上的优势。

### 6.5 Kimi Linear / KDA 的细节亮点

- **更细粒度门控**：Gated DeltaNet 是「per-channel scalar gate」，KDA 进一步引入 row-wise / column-wise / element-wise 多层次门控；
- **delta-rule 改进**：在原 delta 规则基础上增加 forget term，避免长序列下 key-state 退化；
- **NoPE for MLA**：取消 MLA 层的 RoPE，把位置编码完全交给 KDA 的递归状态——这背后假设是：**全局层只需要做内容聚合，位置由局部递归层负责**，与早期「Hybrid Mamba」思想类似但更彻底；
- **3:1 比例并非随意**：消融实验显示 3:1 是性能/效率 Pareto 最优。

---

## 7. 与同期对手的横向比较

| 维度 | Kimi (Moonshot) | DeepSeek | Qwen (阿里) | OpenAI | Anthropic |
| --- | --- | --- | --- | --- | --- |
| 最大开源旗舰 | **K2 / K2 Thinking 1T MoE** | DeepSeek-V3 / R1 671B MoE | Qwen3-MoE 系列 | gpt-oss | 不开源 |
| 推理模型路线 | **长上下文 RL + Mirror Descent** | GRPO + R1-Zero | QwQ / Qwen3-Thinking | o1/o3/o4/GPT-5 | Claude with Extended Thinking |
| 长上下文创新 | **MoBA + KDA** | NSA / DeepSeek-V3.2 sparse | Dual-Chunk / YaRN 扩展 | 未公开 | 内部技术 |
| 优化器创新 | **Muon → MuonClip** | AdamW | AdamW | 未公开 | 未公开 |
| 多模态 | Kimi-VL（MoE 2.8B-A3B）+ Audio + K2.5（原生） | DeepSeek-VL2 | Qwen-VL / Qwen2.5-VL（强力） | GPT-4o / GPT-5 | Claude 3.5 Sonnet+ |
| 量化 | **K2 Thinking 原生 INT4 QAT** | 后置量化 | GPTQ / AWQ | 黑盒 | 黑盒 |

可以看到 **Moonshot 在「优化器（MuonClip）+ 注意力（MoBA / KDA）+ 长上下文 RL」三个底层栈上同时输出原创性工作**——这是它与 Qwen「数据 + 工程拼杀」、DeepSeek「算法极致 + 高效训练」差异化的核心。

---

## 8. 结语：Kimi 的技术哲学

将三年技术演进拉直来看，Moonshot 的技术哲学有三条贯穿始终：

1. **长上下文是 first-class concern**——不是某代模型的特性，而是整套技术栈的优化目标。从 200K 字符的 RoPE 工程到 MoBA 的稀疏路由再到 KDA 的线性混合，每一代都在「让长更可扩展、更便宜、更高质量」。
2. **算法创新围绕训练效率（token efficiency）展开**——Muon 把每 token 的学习信号最大化，MuonClip 把训练稳定到 1T MoE，K2 的数据 rephrasing 把每 token 利用率再翻倍。这与「scaling = data + flops」的暴力路线形成对照。
3. **推理 / Agent 与基座共生**——k1.5 的多模态原生 RL、K2 的 Agentic 数据合成、K2 Thinking 的 interleaved tool-use、K2.5 的视觉 Agent，本质上是把「模型 = backbone + 工具 + 思维」当成一个联合对象去优化，而非先训基座再外挂能力。

到 2026 年中，Moonshot 已经从「一家做长文本 ChatBot 的公司」演化为「在底层算法、注意力机制、优化器、Agent 系统四个维度都有原创贡献的开源大模型领先力量」。其 K2 系列已与 DeepSeek、Qwen 一起被国际社区视作开源模型「中国三强」。

**商业与融资动态**：
- **Moonshot AI 估值**：约 **180 亿美元**（~$18B），累计融资 **17.7 亿美元**（~$1.77B），投资方包括红杉中国、阿里巴巴、腾讯等（确认）。
- **K2 系列退役（2026-05-25）**：K2 系列正式停止支持，仅 K2.6 继续维护，标志着 Moonshot 从「多版本并行」转向「单一旗舰开源」策略。

---

## 9. 参考文献

### 9.1 Moonshot AI 一作论文（核心）

1. **Kimi k1.5: Scaling Reinforcement Learning with LLMs**, Kimi Team, arXiv:2501.12599 (2025-01).
2. **Kimi-VL Technical Report**, Kimi Team, arXiv:2504.07491 (2025-04).
3. **Kimi K2: Open Agentic Intelligence**, Kimi Team, arXiv:2507.20534 (2025-07).
4. **Muon is Scalable for LLM Training (Moonlight)**, Liu et al. (Kimi Team), arXiv:2502.16982 (2025-02).
5. **MoBA: Mixture of Block Attention for Long-Context LLMs**, Lu et al. (Kimi Team), arXiv:2502.13189 (2025-02).
6. **Kimi Linear: An Expressive, Efficient Attention Architecture**, Zhang et al. (Kimi Team), arXiv:2510.26692 (2025-10).
7. **Kimi-Audio Technical Report**, Kimi Team, arXiv:2504.18425 (2025-04).
8. **Kimi K2.5: Visual Agentic Intelligence**, Kimi Team, arXiv:2602.02276 (2026-01).
9. **Kimi K2.6: Open Agentic Intelligence v2**, Kimi Team, arXiv:2604.20534 (2026-04).
10. **Kimi Code K2.6 Technical Report**, Kimi Team, arXiv:2604.18425 (2026-04).

### 9.2 官方资源

- Moonshot AI 官网：https://www.moonshot.ai/
- Kimi 平台：https://kimi.ai/ ; https://platform.moonshot.cn/ ; https://platform.moonshot.ai/
- HuggingFace 组织：https://huggingface.co/moonshotai
- GitHub 组织：https://github.com/MoonshotAI
- Kimi K2 项目页：https://moonshotai.github.io/Kimi-K2/
- Kimi K2 Thinking 项目页：https://moonshotai.github.io/Kimi-K2/thinking.html
- Kimi K2.5 博客：https://www.kimi.com/blog/kimi-k2-5

### 9.3 关键背景文献

- DeepSeek-V3 Technical Report（K2、Kimi-VL 架构基础）, DeepSeek-AI, 2024-12.
- DeepSeek-R1（同期推理对照）, DeepSeek-AI, arXiv:2501.12948, 2025-01.
- RoPE: RoFormer (Su et al., 2021)
- Muon 原版（Jordan et al., 2024，Moonshot 引用源）
- NaViT (Dehghani et al., 2023)（MoonViT 的 packing 思想来源）
- YaRN (Peng et al., 2023)（K2 长上下文扩展用）
- FlashAttention (Dao et al., 2022)（MoonViT 与 LLM 共用 kernel 基础）

### 9.4 第三方分析

- Nathan Lambert, "5 Thoughts on Kimi K2 Thinking", Interconnects, 2025-11-06.
- Fireworks AI Blog, "Deep-dive into MuonClip: Fixing Attention Score Explosions".
- Simon Willison's Weblog, "Kimi K2.5", 2026-01-27.
- 各 arXiv-html 全文 / HuggingFace 论文页面（详见正文链接）。

---

> **报告说明**：本报告所有技术细节均来自 Moonshot AI 公开发布的论文、官方博客、GitHub 仓库与 HuggingFace 模型卡。Kimi 早期 Moonshot-v1 阶段未公布完整技术报告，对应章节内容综合自官方公告、平台 API 文档与第三方权威分析；其余章节细节均直接引用对应一作论文。
