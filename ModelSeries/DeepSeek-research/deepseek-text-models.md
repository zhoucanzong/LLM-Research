# DeepSeek 纯文本模型系列发展脉络

> 本报告深度梳理 DeepSeek 纯文本（基础/通用对话）大语言模型系列的演进路线，从 2024 年 1 月的初代 DeepSeek‑LLM 67B 到 2026 年 4 月发布的 DeepSeek‑V4，覆盖每一代的架构参数、核心创新（MLA、DeepSeekMoE、FP8 训练、MTP、auxiliary‑loss‑free 负载均衡、DSA 稀疏注意力等）、训练数据规模与训练成本，并附 MLA / DeepSeekMoE 专题深度解析、跨代架构对比表和论文引用列表。本报告聚焦 V/V3.x/V4 主线纯文本模型；DeepSeek‑R1/R1‑0528、DeepSeek‑Math、DeepSeek‑Coder、DeepSeek‑VL 等衍生分支只在涉及主线时简略提及。

---

## 0. 时间线总览

| 时间 | 模型 | 类型 | 关键创新 |
| --- | --- | --- | --- |
| 2024-01-05 | **DeepSeek‑LLM 7B / 67B（Base & Chat）** | Dense Transformer | 自研 scaling laws；2T tokens；67B Chat 优于 LLaMA‑2 70B 与 GPT‑3.5 |
| 2024-01-11 | DeepSeekMoE 16B / 145B (研究) | MoE 预研 | 细粒度专家分割 + 共享专家隔离 |
| 2024-05-07 | **DeepSeek‑V2 / V2‑Lite** | MoE | **首次引入 MLA + DeepSeekMoE**；236B/21B 激活；128K 上下文 |
| 2024-05-17 | DeepSeek‑V2‑0517 | MoE | 指令跟随能力大幅提升（IFEval 63.9→77.6） |
| 2024-06-28 | DeepSeek‑V2‑0628 | MoE | 编程、数学、推理能力增强 |
| 2024-09-05 | **DeepSeek‑V2.5** | MoE | DeepSeek V2 Chat 与 DeepSeek‑Coder V2 合并 |
| 2024-12-10 | DeepSeek‑V2.5‑1210 | MoE | 数学/代码进一步提升（MATH‑500 74.8→82.8） |
| 2024-12-26 | **DeepSeek‑V3** | MoE | 671B/37B 激活；FP8 混合精度训练；MTP；auxiliary‑loss‑free 负载均衡；14.8T tokens；2.788M H800 GPU 小时（约 $5.576M） |
| 2025-01-20 | DeepSeek‑R1 | 推理模型 | RLVR + GRPO，与 V3 同架构 |
| 2025-03-24 | DeepSeek‑V3‑0324 | MoE | MMLU‑Pro/AIME/LiveCodeBench 均显著提升 |
| 2025-05-28 | DeepSeek‑R1‑0528 | 推理模型 | AIME 2025 70→87.5 |
| 2025-08-21 | **DeepSeek‑V3.1**（含 V3.1‑Base） | 混合推理 MoE | **混合推理架构**（一个模型支持思考/非思考双模式）；Agent 能力大幅增强 |
| 2025-09-22 | DeepSeek‑V3.1‑Terminus | 修订 | 中英文混杂、Agent 体验修复 |
| 2025-09-29 | **DeepSeek‑V3.2‑Exp** | MoE+稀疏注意力 | **首次引入 DeepSeek Sparse Attention (DSA)**（lightning indexer + token selector） |
| 2025-11-27 | DeepSeekMath V2 | 数学专用 | 自验证（self‑verification）+ 自精炼（self‑refinement）+ meta‑verifier |
| 2025-12-01 | **DeepSeek‑V3.2** | MoE+稀疏注意力 | 与 V3.2‑Exp 同架构；GRPO 更新；可达 GPT‑5 水平 |
| 2025-12-01 | DeepSeek‑V3.2‑Speciale | 扩展思考 MoE | 仅推理数据 RL，长 CoT；2025 IMO/IOI 金牌 |
| 2025-12-31 | mHC（研究） | 残差路径 | Manifold‑Constrained Hyper‑Connections |
| 2026-04-24 | **DeepSeek‑V4‑Pro / V4‑Flash** | MoE+稀疏注意力 | V4‑Pro 1.6T/49B，V4‑Flash 284B/13B；**Token‑wise 压缩 + DSA**；**1M 上下文成为默认** |

DeepSeek 纯文本模型主线演进可总结为四条主线：
1. **架构线**：Dense（V1）→ MoE+MLA+DeepSeekMoE（V2）→ MoE+MLA+DeepSeekMoE+ALF+MTP（V3）→ +DSA（V3.2/V3.2‑Exp）→ Token‑wise 压缩+DSA（V4）。
2. **训练效率线**：BF16（V1/V2）→ FP8 混合精度 + DualPipe + MTP（V3）→ 进一步引入稀疏注意力以降低 long‑context 训练/推理成本（V3.2/V4）。
3. **能力线**：基础模型（V1/V2/V3）→ 推理模型分叉（R1）→ 混合推理（V3.1/V3.2/V4）→ 自验证强化（DeepSeekMath V2 / V3.2‑Speciale）。
4. **上下文线**：4K（V1）→ 128K（V2 起）→ 1M 默认（V4）。

---

## 1. DeepSeek‑LLM (2024‑01)：从 scaling laws 出发的初代 Dense 模型

### 1.1 论文信息
- **标题**：*DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*
- **arXiv**：[2401.02954](https://arxiv.org/abs/2401.02954)（2024‑01‑05 提交）
- **代码 / 权重**：`github.com/deepseek-ai/DeepSeek-LLM`，HuggingFace `deepseek-ai/deepseek-llm-67b-base / chat`、`deepseek-llm-7b-base / chat`

### 1.2 模型架构与规模
- 模型尺寸：**7B 与 67B 两档 Dense Transformer**（与 LLaMA 2 同代但非同构）。
- Tokenizer：基于 BBPE（Byte‑level BPE），**词表 100K**——这一 tokenizer 之后被 V2、V3 全部沿用。
- 上下文长度：4K（初代）。
- Pre‑norm Transformer + RoPE 位置编码 + SwiGLU + GQA（Grouped‑Query Attention，仅在 67B 上）。

### 1.3 训练数据与方法
- **预训练语料：2 万亿 (2T) tokens**，并强调“持续扩展”。
- 数据构建：基于自研去重、过滤、混合策略（中文比例较高）。
- 学习率调度：采用 **multi‑step LR scheduler**（不同于业界常用 cosine）以方便 continual pretraining。
- 后训练：SFT + DPO，得到 DeepSeek‑LLM 7B / 67B Chat。

### 1.4 核心贡献：自研 Scaling Laws
论文最大贡献并不在某个模型本身，而在于**对开源 scaling laws 的再校准**：
1. 给出 **IsoFLOP scaling laws** 在不同数据/模型规模下的拟合，纠正了之前 Chinchilla / Hoffmann 论文中的部分系数。
2. 提出基于 **non‑embedding FLOPs/token (M)** 的最优分配公式，而不是直接用参数量 N 与 token 数 D。
3. 实证：在自研 scaling laws 指引下用 7B 与 67B 两档训练，效果优于业界同规模模型，为后续 V2/V3 奠定方法论基础。

### 1.5 性能要点
- **DeepSeek‑LLM 67B 在多数中英文基准上超过 LLaMA‑2 70B**，尤其在代码、数学、推理三大类。
- 67B Chat 在开放式评估（MT‑Bench 等）上**优于 GPT‑3.5**（67B Chat 系统提示版 8.58）。

---

## 2. DeepSeekMoE (2024‑01)：MoE 架构的方法论铺垫

虽然 DeepSeekMoE 16B/145B 不属于"纯文本主力模型"主线，但其论文（[arXiv:2401.06066](https://arxiv.org/abs/2401.06066)）确立了之后 V2/V3 沿用的两大设计原则：

1. **Fine‑grained Expert Segmentation（细粒度专家分割）**：将专家维度切得更细（例如把每个常规专家拆成 m 份小专家，同时把激活的专家数也乘以 m），获得更强的组合能力与更专业化的知识划分。
2. **Shared Experts Isolation（共享专家隔离）**：保留少量"共享专家"对所有 token 始终激活，吸收公共知识，避免 routed experts 之间的知识冗余。

这两点直接成为 V2/V3 MoE FFN 的标准范式。

---

## 3. DeepSeek‑V2 (2024‑05)：MLA + DeepSeekMoE 范式确立

### 3.1 论文信息
- **标题**：*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*
- **arXiv**：[2405.04434](https://arxiv.org/abs/2405.04434)（v1 2024‑05‑07）
- **HF 模型**：`deepseek-ai/DeepSeek-V2`、`DeepSeek-V2-Chat`、`DeepSeek-V2-Lite(-Chat)`

### 3.2 模型规模与架构
- **总参数 236B，激活参数 21B / token**（每 token 通过 8/8 路 routed + 2 共享专家激活）。
- **DeepSeek‑V2‑Lite**：15.7B 总参 / 2.4B 激活，开源以支持社区研究。
- 上下文长度：**128K**（YaRN 扩展，scale=40, α=1, β=32, target=160K，并对 attention entropy 调节因子 √t = 0.0707·ln s + 1）。
- 基本组件：Pre‑norm + RMSNorm + RoPE + SwiGLU；保留 V1 BBPE 100K tokenizer。

### 3.3 核心创新一：Multi‑head Latent Attention (MLA)
**目标**：在保持 ≥ MHA 性能的前提下大幅压缩 KV cache。

**机制**（V2 论文 §2.1）：
1. 用一个低秩降投影 $W^{DKV}$ 把隐状态 $\mathbf{h}_t$ 压成一个**潜向量 $\mathbf{c}_t^{KV}$**（维度 $d_c$，V2 中 $d_c = 4 d_h$）。
2. 推理时**只缓存 $\mathbf{c}_t^{KV}$**（每 token 仅一份，不分头），需要时再通过 $W^{UK}, W^{UV}$ 上投回完整 K、V。
3. 为兼容 RoPE，引入"解耦 RoPE"分支：单独维护一个共享的小维度 $\mathbf{k}_t^R$（维度 $d_h^R = d_h/2$），承担位置信息；query 端则把 q 拆成 $[q^C; q^R]$，前者来自压缩空间，后者经过 RoPE。
4. 对 query 同样做低秩压缩（仅训练阶段省显存，推理阶段不影响 KV cache）。

**KV cache 大小对比**（V2 论文 Table 1）：

| 注意力机制 | 每 token KV cache 元素数 | 能力 |
| --- | --- | --- |
| MHA | $2 n_h d_h l$ | Strong |
| GQA (g 组) | $2 n_g d_h l$ | Moderate |
| MQA | $2 d_h l$ | Weak |
| **MLA (DeepSeek‑V2)** | $(d_c + d_h^R) l \approx \tfrac{9}{2} d_h l$ | **Stronger** |

也就是说，MLA 的 KV cache 体积**约等于 GQA 仅 2.25 组**的水平，但效果**强于 MHA**——这是 V2 能把 128K 上下文做到推理可负担的关键。

**与 GQA/MQA 的对比要点**：
- GQA/MQA 是"减少 KV 头数"，是结构性削减信息容量。
- MLA 通过低秩 joint compression 在**信息维度**上压缩，并保留独立的 RoPE 通道，使得关键的位置信息不被压缩破坏，因此能在更小 cache 下取得更优性能。
- 推理时由于 $W^{UK}$ 可以被吸收进 $W^Q$，整个上投影可以省掉一次额外乘法（"absorb" 技巧），仅在 RoPE 路径上保留小开销。

### 3.4 核心创新二：DeepSeekMoE FFN
- 路由专家数量大、激活数也较大（V2：每层 160 个 routed experts + 2 个 shared experts，每 token 激活 6 个 routed experts）。
- **Device‑limited routing**：限制每个 token 的目标专家最多落在 M 台设备上（M ≥ 3 即可达到接近无限制 top‑K 的效果），降低跨节点 all‑to‑all。
- **三层辅助损失**：
  - 专家级负载均衡 $\mathcal{L}_{\text{ExpBal}}$；
  - 设备级负载均衡 $\mathcal{L}_{\text{DevBal}}$；
  - 通信均衡 $\mathcal{L}_{\text{CommBal}}$。
- **Token‑Dropping 策略**：训练效率优化（论文 §2.2.4）。

### 3.5 训练数据与基础设施
- 预训练语料：**8.1T tokens**（中文比英文多约 12%），相比 V1 数据量更大、质量更高。
- SFT：1.5M 多领域对话；RL：**GRPO**（来自 DeepSeekMath，2024）。
- 训练框架：**HAI‑LLM**，16‑way zero‑bubble 流水线并行 + 8‑way 专家并行 + ZeRO‑1；**未使用 tensor 并行**（因激活参数少 + 部分算子重计算）；定制 CUDA kernel 与改造版 FlashAttention‑2。
- 集群：**NVIDIA H800**，节点内 NVLink/NVSwitch，节点间 InfiniBand。
- **训练成本**：每训练 1T tokens，DeepSeek 67B 需 300.6K GPU‑hours，**DeepSeek‑V2 仅需 172.8K**——稀疏 V2 比稠密 67B **节省 42.5% 训练成本**，同时推理 FP8 + 6‑bit KV cache 量化使最大吞吐相比 67B 提升 5.76×。

### 3.6 V2 系列后续小版本
- **V2‑0517**（2024‑05‑17）：IFEval Prompt‑level acc 63.9% → 77.6%；优化 system prompt。
- **V2‑0628**（2024‑06‑28）：HumanEval 79.88→84.76；MATH 55.02→71.02；BBH 78.56→83.40；Arena‑Hard vs GPT‑4‑0314 胜率 41.6→68.3。
- **V2.5**（2024‑09‑05）：DeepSeek V2 Chat 与 DeepSeek‑Coder V2 **合并为单一模型**，同时通过 `deepseek-coder` / `deepseek-chat` 两个 API 名访问。
- **V2.5‑1210**（2024‑12‑10）：MATH‑500 74.8→82.8；LiveCodeBench(0801–1201) 29.2→34.38；优化文件上传与网页摘要。

---

## 4. DeepSeek‑V3 (2024‑12)：FP8、MTP、Auxiliary‑Loss‑Free，5.576M 美元的极致性价比

### 4.1 论文信息
- **标题**：*DeepSeek‑V3 Technical Report*
- **arXiv**：[2412.19437](https://arxiv.org/abs/2412.19437)（v1 2024‑12‑27，v2 2025‑02‑18）
- **HF 模型**：`deepseek-ai/DeepSeek-V3`、`DeepSeek-V3-Base`
- API：2024‑12‑26 起 `deepseek-chat` 升级到 V3。

### 4.2 模型规模
- **总参数 671B，激活 37B / token**。
- 沿用 MLA + DeepSeekMoE，但专家数与维度大幅扩张（V3 每层 1 个共享专家 + 256 个 routed experts，每 token 激活 8 个 routed experts；MoE 层共 61 层，前几层为 dense FFN）。
- 上下文长度：128K（仍用 YaRN）。

### 4.3 核心创新

#### 4.3.1 Auxiliary‑Loss‑Free 负载均衡
- V2 中的负载均衡完全由辅助损失（ExpBal/DevBal/CommBal）驱动，但**辅助损失会损害模型主任务性能**。
- V3 引入 **per‑expert bias**：在 routing softmax 中给每个专家加一个动态调整的 bias，**该 bias 不参与梯度反传**，仅根据近期的负载情况以小步长加减。负载偏低的专家把 bias 调高、偏高的调低，从而**纯靠 bias 调度实现均衡，几乎不再依赖辅助 loss**——只保留极小权重的 sequence‑wise auxiliary loss 作为最后保险。
- 效果：负载均衡和模型质量同时改进，论文给出消融。

#### 4.3.2 Multi‑Token Prediction (MTP)
- 训练目标在主 next‑token loss 之外，**额外预测下一 D 个 token**（V3 中 D=1，即除当前位置外再多预测 1 个 token）。
- 实现上是在主模型之上挂载一个**轻量级 MTP 模块**（一层 transformer + projection），共享主干 embedding/output head，在训练阶段只增加少量 FLOPs。
- 两点收益：
  1. **训练信号更密**，主模型本身在所有标准基准上得到稳定提升；
  2. 推理时可作为**投机解码（speculative decoding）的 draft module**，论文报告**第二 token 接受率约 85–90%**，端到端 TPS 大约 ×1.8。
- 这是大型 MoE 上首次将 MTP 工业级落地。

#### 4.3.3 FP8 混合精度训练
V3 是**首个在 671B 量级 LLM 上跑通 FP8 端到端训练**的开源模型。技术细节：
- 大部分 GEMM（Wgrad/Dgrad/Fwd）使用 **FP8 E4M3**；BF16/FP32 仅保留在 norm、loss、master weight、optimizer state、attention softmax 等敏感部分。
- **Tile‑wise / Block‑wise fine‑grained 量化**：对激活按 1×128 tile 量化、对权重按 128×128 block 量化，避免 tensor‑wise scaling 在 outlier 下溢出。
- **高精度累加**：FP8 GEMM 的部分和按周期升级到 FP32 累加（修补了 Hopper 架构 FP8 累加器精度不足的问题）。
- **统一 E4M3**：与 NVIDIA 默认配方（forward E4M3 + backward E5M2）不同，V3 全用 E4M3，简化框架并配合 fine‑grained 量化保证数值稳定。
- 收益：相比 BF16 训练**显存减半 + GEMM 算力翻倍**，没有任何 loss spike 或回滚。

#### 4.3.4 DualPipe 流水线并行
- 自研 **DualPipe** 算法：让前向和反向同时双向流动，**几乎完全重叠 all‑to‑all 通信与计算**。
- 配合 cross‑node MoE 通信优化（IB + NVLink 混合 all‑to‑all kernel），使得 H800 上跨节点专家并行的开销可忽略。

### 4.4 训练数据与成本
- 预训练：**14.8T 多源高质量 tokens**（比 V2 的 8.1T 提升 ~83%），中英文以外强化代码与数学。
- 后训练：SFT + RL（继续使用 GRPO，蒸馏部分知识来自 R1）。
- **训练成本**（论文摘要原话）：
  - 总计 **2.788M H800 GPU‑hours**；
  - 按 H800 租金 \$2/hour 折算 ≈ **\$5.576M**；
  - 全过程**未发生不可恢复的 loss spike，未做任何回滚**。
- 该成本仅含主训练 GPU 算力，**不包含数据、消融实验、人力等成本**——这一点 DeepSeek 团队也在多个采访中澄清过。

### 4.5 性能
- 在 MMLU‑Pro、GPQA、MATH‑500、AIME、LiveCodeBench、Codeforces 等基准上**全面超越所有开源模型**，与 GPT‑4o、Claude 3.5 Sonnet 等闭源旗舰相当或互有胜负。
- API 升级时间线：
  - **V3‑0324**（2025‑03‑24）：MMLU‑Pro 75.9→81.2，GPQA 59.1→68.4，AIME 39.6→59.4，LiveCodeBench 39.2→49.2，前端代码生成与中文写作显著改善。
- 同架构衍生：**DeepSeek‑R1**（2025‑01‑20）以 V3‑Base 为底座，用 RLVR + GRPO 训练出推理模型，是 2025 年最具影响力的开源推理模型之一。**DeepSeek‑R1‑0528**（2025‑05‑28）将 AIME 2025 从 70.0 推到 87.5（+17.5），GPQA 71.5→81.0，Aider 57.0→71.6。

---

## 5. DeepSeek‑V3.1 (2025‑08)：混合推理架构

### 5.1 关键变化
- 2025‑08‑21 通过 API 上线，`deepseek-chat` = V3.1 非思考模式，`deepseek-reasoner` = V3.1 思考模式——**用同一个 checkpoint 通过 prompt template 切换**，开启 DeepSeek 的"hybrid reasoning"路线（与 Qwen3 早期版本类似）。
- 架构与 V3 / V3‑Base 一致，差异完全在后训练数据与训练流程；DeepSeek 同时放出 **V3.1‑Base** 让社区做自定义后训练。
- 推理效率提升：相比 R1‑0528，V3.1‑Think 给出同等答案所用 token 数显著降低。
- Agent 能力大幅增强：
  - SWE‑bench Verified **66.0**
  - SWE‑bench Multilingual **54.5**
  - Terminal‑bench **31.3**
- **V3.1‑Terminus**（2025‑09‑22）：修复中英混杂、异常字符，进一步优化 Code Agent / Search Agent 表现，是 V3.2 的训练 base。

---

## 6. DeepSeek‑V3.2 系列 (2025‑09 ~ 2025‑12)：DeepSeek Sparse Attention (DSA)

### 6.1 V3.2‑Exp (2025‑09‑29)
- 实验性版本，目的之一是**让推理基础设施提前适配 DSA**。
- 在 V3.1‑Terminus 之上 **continued training** 引入 **DeepSeek Sparse Attention (DSA)**——一种细粒度可学习稀疏注意力。
- 685B 总参数（沿用 V3 系列规模）。

### 6.2 DeepSeek Sparse Attention (DSA) 原理
DSA 由两部分构成：

1. **Lightning Indexer**——为每个 query token 计算其对所有历史 token 的"相关度评分"。在 MLA 压缩空间中复用 keys，避免重复存储；并使用 ReLU 与 per‑head 学习权重的简单形式：
   $$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}\, \mathrm{ReLU}(q_{t,j} \cdot k_s)$$
   其中 t 是当前 query 位置，s 是历史 token 位置（s<t），j 是 indexer head 索引；keys 直接从 MLA 已压缩并缓存的 latent 取出。

2. **Token Selector**——基于 indexer 分数选 **top‑k**（DeepSeek 公开代码中 k=2048），构造稀疏 attention mask，只对被选中的历史 token 计算 attention。

效果：
- 注意力复杂度从 $O(L^2)$ 降为 **$O(L \cdot k)$**（k≪L），对长上下文的训练与推理成本大幅下降；
- 设计目标不是"超过 V3.1‑Terminus"，而是在长上下文上**几乎不损失质量**的前提下取得效率优势。

### 6.3 DeepSeekMath V2 (2025‑11‑27)
基于 V3.2‑Exp‑Base 的数学专用模型，2025 IMO 与 IOI 金牌级别。提出两大方法被回流到 V3.2：
- **Self‑Verification**：训练独立的 LLM verifier（LLM 2）对 generator（LLM 1）输出打分，再训练 meta‑verifier（LLM 3）评估 verifier；rubric 评分为 1/0.5/0。
- **Self‑Refinement**：训练好的 generator 学会按 rubric 对自己的输出迭代修正（最多 8 轮，仍未饱和）。

### 6.4 V3.2 (2025‑12‑01)
- arXiv [2512.02556](https://arxiv.org/abs/2512.02556)：*DeepSeek‑V3.2: Pushing the Frontier of Open Large Language Models*
- 架构与 V3.2‑Exp 完全相同（MLA + DeepSeekMoE + DSA），但训练流程升级：
  - **可扩展 RL 框架**：post‑training 算力扩大，性能与 GPT‑5 相当。
  - **大规模 Agentic Task 合成 pipeline**：系统化生成 tool‑use 训练数据。
  - **GRPO 升级**：保留 KL（按域调权，数学域近 0）、修正 KL 估计、off‑policy 序列掩码、MoE rollout 路由保持、top‑p/top‑k 采样掩码保持、保留原 GRPO normalization。
  - **奖励体系**：可验证域用规则化 outcome reward + 长度惩罚 + 语言一致性；不可验证域用生成式 reward model（按 prompt 维度指定 rubric）；数学域吸收 DeepSeekMath V2 的数据与 reward 方法。
- **V3.2‑Speciale**：仅推理数据 RL、放宽长度惩罚的"扩展思考"变体，IMO 2025 与 IOI 2025 双金；推理 token 数显著高于 V3.2。

---

## 7. DeepSeek‑V4 (2026‑04‑24)：1M 上下文成为标配

来源：DeepSeek API 官方公告（[news260424](https://api-docs.deepseek.com/news/news260424)）与 HuggingFace `deepseek-ai/DeepSeek-V4-Pro` 的技术报告 PDF。

### 7.1 双模型策略
| 子模型 | 总参 / 激活 | 定位 |
| --- | --- | --- |
| **DeepSeek‑V4‑Pro** | **1.6T / 49B** | 旗舰，对标 GPT/Gemini/Claude 闭源旗舰；开源 Agentic Coding SOTA |
| **DeepSeek‑V4‑Flash** | **284B / 13B** | 高性价比，推理能力接近 Pro，简单 Agent 任务持平 Pro |

均支持双模式（思考 / 非思考），均通过 OpenAI ChatCompletions 与 Anthropic 接口提供。

### 7.2 架构创新
- **新型注意力 = Token‑wise 压缩 + DSA**——把 V3.2 的 DSA 与一种 token‑wise 隐空间压缩组合，进一步降低长上下文的算力与显存。
- **1M 上下文成为所有 DeepSeek 官方服务的默认上限**（API max=384K 输出/上下文细节见 V4 文档）。
- 与 Claude Code、OpenClaw、OpenCode 等 agent 生态无缝对接，已在 DeepSeek 内部用于 agentic coding。
- 模型权重通过 HuggingFace `deepseek-ai/DeepSeek-V4-Pro / V4-Flash` 集合开源。

> ⚠️ 与官方 changelog 一致：自 2026‑07‑24 起 `deepseek-chat` / `deepseek-reasoner` 旧名将退役，期间分别路由至 `deepseek-v4-flash` 的非思考 / 思考模式。

---

## 8. MLA 专题深度解析

### 8.1 KV cache 是 LLM 推理的真正瓶颈
对一个 N 层、$n_h$ 头、每头 $d_h$ 维、序列长度 L 的 MHA，KV cache 大小为 $2 N n_h d_h L \cdot \text{bytes}$。在 100B+ 模型 + 128K 上下文场景下，单 batch KV cache 通常以**百 GB**计，是 prefill/decoding 的主导开销。

### 8.2 三种历史方案
| 方案 | 思路 | 缺点 |
| --- | --- | --- |
| MHA | 每头独立 K/V | KV 巨大 |
| MQA (Shazeer 2019) | 所有头共享一组 K/V | KV 极小但损质量明显 |
| GQA (Ainslie 2023) | 把头分 g 组，每组共享 K/V | 需在 g 上做权衡，超长上下文下仍偏大 |

### 8.3 MLA 的核心三步
1. **Joint low‑rank compression**：$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t$，K/V 共用一份压缩潜向量。
2. **延迟上投影**：$\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}$，$\mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}$；得益于矩阵结合律，$W^{UK}$ 在推理时可吸收进 $W^Q$，从而**只需缓存 $\mathbf{c}_t^{KV}$**（极小）即可计算 attention。
3. **解耦 RoPE 通道**：因 RoPE 与低秩压缩不可交换（合并矩阵会被 RoPE 阻断），单独维护一个共享小维度的 RoPE 路径 $\mathbf{k}_t^R$ 与 query 拼接 $[q_t^C; q_t^R]$、key 拼接 $[k_{t,i}^C; k_t^R]$，使位置信息独立维护、不破坏低秩 absorb 优化。

### 8.4 MLA vs GQA/MQA：信息论视角
- GQA 是**结构性容量裁剪**：直接减少 K/V 头数，等价于让所有头共用同一 subspace，损失多视角能力。
- MLA 是**信息论级压缩**：压缩到 $d_c$ 维潜空间，但通过解耦 RoPE 与 absorb 技巧，**模型参数容量与计算能力并未削弱**——KV cache 减少来自"潜空间储存 + 推理时上投回原空间"，类似 LoRA 的 down‑project / up‑project 之于权重。这也是 V2 论文 Appendix D 中 MLA 在同 KV 体积下质量优于 MHA 的根因。

### 8.5 V3 / V3.2 / V4 中 MLA 的演化
- V2 → V3：MLA 结构不变，但通过 FP8 + DualPipe 把训练效率推到极致；MLA 的 KV cache 进一步以 6‑bit 量化部署。
- V3 → V3.2：MLA 之上叠加 **DSA**，注意力复杂度从 $O(L^2)$ 降到 $O(L k)$；DSA 的 lightning indexer 直接复用 MLA 已压缩的 key latent，避免重复缓存。
- V3.2 → V4：在 DSA 之上再引入 token‑wise 压缩（公告原文 *Novel Attention: Token-wise compression + DSA*），把 1M 默认上下文做到可负担。

---

## 9. DeepSeekMoE 专题深度解析

### 9.1 标准 MoE 的两个痛点
- **专家专业化不足**：经典 GShard/Switch Transformer 把 FFN 切成几十到一百多个专家，每个专家仍然学到大量"通用知识"，专家间冗余高。
- **routed 专家承担公共知识**导致负载更难均衡，且容易出现"路由塌缩"。

### 9.2 DeepSeekMoE 的两大设计
1. **Fine‑grained Expert Segmentation**：保持 FLOPs 不变，把每个专家切成 m 等份的小专家，同时把激活的专家数量也乘以 m。这意味着在不增加每 token 计算量的前提下，**专家组合空间从 $\binom{N}{K}$ 扩张到 $\binom{mN}{mK}$**，专业化粒度大幅提升。
2. **Shared Experts Isolation**：拿出极少量（V2 中 2 个、V3 中 1 个）"共享专家"对所有 token 始终激活，承担"公共语言/通用知识"，让 routed experts 把容量留给真正需要专业化的方向。

### 9.3 V2 阶段的负载均衡（多重辅助损失 + device‑limited routing）
- ExpBal：基于路由频率 $f_i$ 与平均概率 $P_i$ 的乘积惩罚不均衡。
- DevBal：把专家划分为 D 组（每组一台设备），按设备维度计算同样形式的损失。
- CommBal：对接收侧通信量做均衡，配合 device‑limited routing（每 token 只能落到 ≤ M 台设备）。
- 还引入 **token‑dropping**：超过设备容量的 token 直接 drop。

### 9.4 V3 阶段：Auxiliary‑Loss‑Free Load Balancing
- 用 **per‑expert bias term** 替代主要辅助损失：$\tilde{s}_{i,t} = s_{i,t} + b_i$，bias 通过移动平均更新，**不参与梯度反传**——既能调度负载又不污染主任务损失。
- 仅保留极小权重的 **sequence‑wise auxiliary loss** 作为防止极端不均衡的保险。
- 这是 V3 同时取得"负载更均衡 + 主任务效果更好"的关键之一。

### 9.5 路由与并行：从 device‑limited 到节点感知
- V2：device‑limited routing（≤ M 设备）。
- V3：node‑limited routing + cross‑node all‑to‑all 优化（IB + NVLink 双层 kernel），**MoE 通信开销几乎被 DualPipe 完全 overlap**。
- 推理：MLA 的小 KV + DeepSeekMoE 的稀疏激活，使得 671B/37B 这种"低激活高总参"模型可以在远小于其总参的显存下高效部署。

---

## 10. 跨代架构对比表

| 维度 | DeepSeek‑LLM (V1, 2024‑01) | DeepSeek‑V2 (2024‑05) | DeepSeek‑V3 / V3.1 (2024‑12 / 2025‑08) | DeepSeek‑V3.2 (2025‑12) | DeepSeek‑V4 (2026‑04) |
| --- | --- | --- | --- | --- | --- |
| 模型类型 | Dense | MoE | MoE | MoE + 稀疏注意力 | MoE + 稀疏注意力 |
| 总参 / 激活 | 7B / 67B (Dense) | 236B / 21B（V2‑Lite 15.7B/2.4B） | 671B / 37B | 685B / ~37B | Pro 1.6T/49B；Flash 284B/13B |
| 注意力 | MHA / GQA | **MLA** | MLA | MLA + **DSA**（lightning indexer + token selector，top‑k≈2048） | **Token‑wise 压缩 + DSA** |
| FFN | SwiGLU Dense | DeepSeekMoE（160 routed + 2 shared，激活 6 routed） | DeepSeekMoE（256 routed + 1 shared，激活 8 routed） | 同 V3 | 升级版 DeepSeekMoE |
| 负载均衡 | — | ExpBal+DevBal+CommBal 三辅助损失 + device‑limited routing | **Auxiliary‑Loss‑Free（per‑expert bias）** + 极小 seq‑wise loss | 沿用 V3 + DSA 友好路由 | 同 V3.x（公开材料未细化） |
| 训练目标 | next‑token | next‑token | **+ Multi‑Token Prediction (D=1)** | + MTP | + MTP（推断） |
| 训练精度 | BF16 | BF16 | **FP8 (E4M3) + 高精度累加 + tile/block 量化** | FP8 | FP8（推断） |
| 流水/通信 | 标准 ZeRO + PP | 16‑way zero‑bubble PP + 8‑way EP + ZeRO‑1 | **DualPipe** + 跨节点 all‑to‑all + ZeRO‑1 | 同 V3 | 同 V3.x |
| 上下文 | 4K | 128K（YaRN） | 128K（V3.1 仍 128K） | 128K | **1M 默认** |
| 预训练 tokens | 2T | 8.1T | 14.8T | 14.8T 之上继续训练 + sparse 训练阶段 | 未公开具体值 |
| 训练成本 (主算力) | — | 每 1T tokens 172.8K H800h | **2.788M H800h ≈ \$5.576M** | 未单独披露 | 未单独披露 |
| 后训练 | SFT + DPO | SFT + GRPO | SFT + RL（含 R1 蒸馏） | RLVR + 生成式 reward + GRPO 升级；吸收 DeepSeekMath V2 自验证/自精炼 | RLVR + Agentic 任务合成 |
| 推理形态 | Base + Chat | Base + Chat | Base + Chat（V3.1 起 hybrid thinking） | hybrid thinking + Speciale 扩展思考 | Pro/Flash + 双模式 |
| 关键开源/API | HF + GitHub | HF + GitHub | HF + GitHub | HF + GitHub | HF + API |

---

## 11. 关键工程要点速查

- **Tokenizer**：BBPE，词表 100K，自 V1 起未变。
- **Position Encoding**：RoPE（V2 起对 RoPE 通道做解耦以兼容 MLA）。
- **Long Context**：YaRN（V2 设置 scale=40, α=1, β=32, target=160K，attention scaling 用 √t = 0.0707·ln s + 1）；V4 直接默认 1M。
- **训练硬件**：V2/V3 均为 H800 集群（节点内 NVLink/NVSwitch + 节点间 IB）；2025 年下半年传闻短暂尝试华为昇腾，但 V3.2 报告显示又回到 NVIDIA（Sebastian Raschka 综述）。
- **训练框架**：自研 **HAI‑LLM**。
- **推理部署**：MLA + 6‑bit KV cache 量化（V2）+ FP8 推理；V3.2/V4 进一步通过 DSA + token‑wise 压缩降本。
- **稳定性**：V3 整个 14.8T tokens 训练**未发生不可恢复 loss spike**，是开源大模型工程实践标杆。

---

## 12. 论文与官方资料引用列表

### 12.1 主线技术报告
1. DeepSeek‑AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*. arXiv:2401.02954, 2024. https://arxiv.org/abs/2401.02954
2. Dai, D. *et al.* *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. arXiv:2401.06066, 2024. https://arxiv.org/abs/2401.06066
3. DeepSeek‑AI. *DeepSeek‑V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv:2405.04434, 2024. https://arxiv.org/abs/2405.04434
4. DeepSeek‑AI. *DeepSeek‑V3 Technical Report*. arXiv:2412.19437, 2024–2025. https://arxiv.org/abs/2412.19437
5. DeepSeek‑AI. *DeepSeek‑V3.2: Pushing the Frontier of Open Large Language Models*. arXiv:2512.02556, 2025. https://arxiv.org/abs/2512.02556
6. DeepSeek‑AI. *DeepSeek‑V4 Technical Report*. HuggingFace, 2026. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf

### 12.2 衍生与配套论文
7. DeepSeek‑AI. *DeepSeek‑R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948, 2025.
8. Shao, Z. *et al.* *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（提出 GRPO）. arXiv:2402.03300, 2024.
9. DeepSeek‑AI. *DeepSeekMath V2*. arXiv:2511.22570, 2025（自验证 / 自精炼）。
10. *Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention*. arXiv:2506.02523, 2025（第三方 MLA 硬件分析）。
11. DeepSeek‑AI. *mHC: Manifold-Constrained Hyper-Connections*. arXiv:2409.19606（更新版 2025‑12‑31 发布）。

### 12.3 官方公告与 changelog
12. DeepSeek API Docs – Change Log. https://api-docs.deepseek.com/updates （包含 2024‑05 ~ 2026‑04 全部 API 升级记录）。
13. DeepSeek API Docs – DeepSeek‑V3.1 Release (2025‑08‑21). https://api-docs.deepseek.com/news/news250821
14. DeepSeek API Docs – DeepSeek‑V3.1‑Terminus (2025‑09‑22). https://api-docs.deepseek.com/news/news250922
15. DeepSeek API Docs – Introducing DeepSeek‑V3.2‑Exp (2025‑09‑29). https://api-docs.deepseek.com/news/news250929
16. DeepSeek API Docs – DeepSeek‑V4 Preview Release (2026‑04‑24). https://api-docs.deepseek.com/news/news260424

### 12.4 第三方综述（用于交叉验证）
17. Sebastian Raschka. *A Technical Tour of the DeepSeek Models from V3 to V3.2*. Ahead of AI Magazine, 2025‑12. https://magazine.sebastianraschka.com/p/technical-deepseek
18. Rohan Paul. *DeepSeek‑V3's Architectural Revolution*. https://www.rohan-paul.com/p/deepseek-v3-technical-report-they

### 12.5 GitHub 资源
- `https://github.com/deepseek-ai/DeepSeek-LLM`
- `https://github.com/deepseek-ai/DeepSeek-MoE`
- `https://github.com/deepseek-ai/DeepSeek-V2`
- `https://github.com/deepseek-ai/DeepSeek-V3`
- `https://github.com/deepseek-ai/DeepSeek-V3.2-Exp`
- `https://huggingface.co/collections/deepseek-ai/deepseek-v4`

---

## 13. 总结：DeepSeek 纯文本主线模型的核心叙事

1. **"用 scaling laws 选定路线"**——V1 用自研 scaling laws 在 7B/67B 上锁定起跑点。
2. **"在架构上抠成本"**——V2 同时引入 MLA（攻 KV cache）与 DeepSeekMoE（攻 FFN），把"低激活高总参"路线一次性确立。
3. **"在训练栈上抠成本"**——V3 用 FP8、DualPipe、auxiliary‑loss‑free、MTP 把 671B 的训练压到 \$5.576M 的纸面成本，并第一次让开源 MoE 与闭源旗舰正面同场。
4. **"长上下文与混合推理常态化"**——V3.1 把推理与对话合并；V3.2 用 DSA 把长上下文 attention 复杂度从 $O(L^2)$ 降到 $O(Lk)$；DeepSeekMath V2 把自验证/自精炼方法回流。
5. **"百万上下文成为默认"**——V4 用 *Token‑wise 压缩 + DSA* 把 1M 上下文做成所有官方服务的默认，并以 Pro/Flash 双模型同时覆盖最高端与高性价比段位。

可以把 V1→V4 看作一条始终围绕"**在同等智能下推理与训练成本系统性降低一个数量级**"的工程路线——MLA、DeepSeekMoE、ALF、MTP、FP8、DualPipe、DSA、Token‑wise 压缩——每一代都把同一个目标推进半步到一步，是开源 LLM 中最具工程纪律的演化序列之一。
