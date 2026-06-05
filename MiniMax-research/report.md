# MiniMax 系列模型研究报告：以 Lightning Attention 重写 Transformer 的扩展曲线

> 撰写时间：2026-06 ｜ 研究范围：MiniMax-abab 系列、MiniMax-01（Text/VL）、MiniMax-Speech 与海螺音视频模型、MiniMax-M1/M2 推理与 Agent 模型、海螺 AI 产品线

---

## 1. 概述与时间线

MiniMax（稀宇科技）成立于 2021 年 12 月，由原商汤科技副总裁、研究院副院长闫俊杰创立，总部位于上海，是中国第一梯队的通用人工智能公司之一。其在大模型路线选择上有两个标志性的早期赌注：（1）**全面押注 MoE（Mixture-of-Experts）**，是国内最早将 MoE 用于万亿参数级文本基座的公司；（2）**全面押注线性注意力（Linear Attention）**，最终在 2025 年初发布 MiniMax-01，用 Lightning Attention 把开源模型的上下文窗口直接拉到 4M token 级别——比同期主流模型长 20–32 倍。MiniMax 的研究矩阵覆盖文本、视觉、语音、音乐、视频五大模态，对外形成 **海螺 AI（含 Hailuo Video、Hailuo Audio）+ Talkie/星野（社交陪伴）+ MiniMax 开放平台 + MiniMax Agent** 多条产品线。

**关键时间线**

| 时间 | 模型 / 事件 | 关键技术 / 形态 |
|------|--------------|------------------|
| 2021-12 | MiniMax 成立 | 闫俊杰创立，立足通用大模型 + 多模态 |
| 2022 末 – 2023 初 | abab 早期版本（abab1/2） | 自研稠密 Transformer 基座 |
| 2023-04 | MiniMax 开放平台正式上线 | API 形态对外提供模型能力 |
| 2023 中后期 | abab5 / abab5.5 | 扩规模、强化指令/对话能力 |
| 2024-01 | abab6 | **国内首个 MoE 大语言模型** |
| 2024-04-17 | abab6.5 / abab6.5s | 万亿参数 MoE，**200k token 上下文** |
| 2024 下半年 | 海螺 AI 视频上线（Video-01、I2V-01-Live） | 多模态产品线开始集中爆发 |
| 2024 下半年 – 2025 初 | T2A-01 / T2V-01-Director / I2V-01-Director | 文生音、镜头控制视频 |
| 2025-01-14 | **MiniMax-Text-01 / MiniMax-VL-01** 论文与权重开源（arXiv:2501.08313） | **456B 总参 / 45.9B 激活** MoE + **Lightning Attention + Softmax 混合注意力**，训练 1M、推理 4M token |
| 2025-05-12 | MiniMax-Speech 技术报告（arXiv:2505.07916） | AR Transformer + 可学习 Speaker Encoder + Flow-VAE，32 语种零样本声纹克隆，登顶 Artificial Arena |
| 2025 上半年 | Speech-02-series、MiniMax Music 1.5 / 02 | 文本到音频、音乐生成 |
| 2025-06-16 | **MiniMax-M1**（arXiv:2506.13585） | **首个开源大规模混合注意力推理模型**，1M 上下文，提出 CISPO RL 算法，3 周 / 53.47 万美元完成 RL 训练 |
| 2025-10 | **MiniMax-M2** 开源 | **230B / 10B 激活** MoE，主打 coding + agent，登顶 Artificial Analysis 开源综合榜 |
| 2026-01-29 | MiniMax Music 2.5 | 段落级控制、音质与可控性大幅提升 |
| 2026-03 前后 | Music 2.5+，MiniMax-M2.5（Agent 生产级） | 全栈编程、工具调用、多步推理 |

整条时间线的演进逻辑可以概括为：**abab 时代押注 MoE → MiniMax-01 时代押注线性注意力 → M1/M2 时代押注 RL + Agent，并把 Lightning Attention 的训练/测试时计算优势迁移到推理模型**。

---

## 2. 基座模型演进：从 abab 到 MiniMax-01 再到 M-series

### 2.1 abab 系列（2022–2024）：押注 MoE 的稠密→稀疏过渡

`abab` 名字来自于 MiniMax 内部对自研基座系列的代号。早期 abab1/2 是常规 Decoder-Only 稠密 Transformer，主要承担产品（Glow / Talkie / 星野）的对话与人设生成能力；abab5 / abab5.5（2023）开始引入更大规模训练数据与强化的对话/指令对齐管线，对中文综合能力做了重点优化。

真正的转折点是 **abab6（2024 年 1 月）**：MiniMax 把 80% 以上的研发精力投在了 MoE 上，发布了**国内首个 MoE 大语言模型**。MoE 的稀疏激活策略让 abab6 在保持单 token 推理成本可控的前提下，把"总知识容量"提升一个数量级。

**abab6.5 / abab6.5s（2024-04-17）** 进一步将 MoE 推到**万亿（trillion）级总参数**，并把上下文窗口扩展到 **200k tokens**：

- `abab6.5`：万亿总参 MoE，200k 上下文，对标 GPT-4 / Claude 2.1，强项是长文档理解、复杂指令、多语种与中文综合能力。
- `abab6.5s`：相同训练数据与方法，但更轻量，推理更快，**1 秒可处理近 3 万字**，主打高吞吐场景。
- 两者通过 MiniMax 开放平台、海螺 AI 等产品对内对外滚动更新。

abab 系列的最大遗留问题，是仍然以**标准 Softmax Attention** 为主，长上下文成本随序列长度二次方上升，到 200k 已逼近"算力可承受 + 工程可调度"的边界。MiniMax 的下一步思路非常清晰：要把上下文做到百万级、千万级，就必须把 Softmax 换掉。

### 2.2 MiniMax-01：把 Lightning Attention 推到 456B 规模

2025 年 1 月 14 日，MiniMax 发布技术报告 *MiniMax-01: Scaling Foundation Models with Lightning Attention*（arXiv:2501.08313），同步在 GitHub 与 Hugging Face 开源 **MiniMax-Text-01 / MiniMax-VL-01** 两个模型权重，许可使用宽松的 MiniMax 协议。

**MiniMax-Text-01 关键架构参数**（来自官方仓库与论文）：

| 维度 | 数值 |
|------|------|
| 总参数 | **456B** |
| 单 token 激活参数 | **45.9B** |
| 层数 | 80 |
| 注意力头数 | 64 |
| 注意力头维度 | 128 |
| Hidden Size | 6144 |
| 词表 | 200,064 |
| MoE 专家数 | 32（FFN 隐层 9216） |
| 路由策略 | Top-2 |
| 位置编码 | RoPE，base=1e7，仅作用于一半头维度 |
| **混合注意力比例** | **每 7 层 Lightning Attention 后接 1 层 Softmax Attention** |
| 训练上下文长度 | 1,000,000 tokens |
| 推理外推上下文 | **4,000,000 tokens**（NIAH 4M 通过） |

这是当时业界第一次把**Linear Attention 真正放到接近半万亿参数 + MoE + 百万级上下文**的工程规模上跑通，并开源完整权重。其核心组合拳是：

1. **Lightning Attention（线性注意力高效实现）**——把训练/推理成本从 O(n²d) 降到 O(nd²)，并通过 IO-aware tiling 把常数项压平；
2. **Softmax 兜底**——只在每 8 层中保留 1 层 Softmax，承担长程检索与精确召回任务，弥补线性注意力在"针在干草堆"等任务上的检索弱项；
3. **MoE 稀疏激活**——保证容量与质量，单 token 仅激活 45.9B 参数；
4. **针对 MoE + Lightning Attention 共同优化的并行策略与 communication-overlap 技术**，让百万级序列的训练 / 推理在数千卡集群上可承担。

在标准基准上，MiniMax-Text-01 与 GPT-4o (11-20)、Claude-3.5-Sonnet、DeepSeek-V3、Llama-3.1-405B 等同期闭源/开源最强模型基本同档（MMLU 88.5、IFEval avg 89.1、Arena-Hard 89.1、C-SimpleQA 67.4 领先），同时在 RULER 1M、LongBench v2、MTOB（半本/整本书翻译）等长上下文任务上明显领先：

- **RULER（1M 长度）**：MiniMax-Text-01 = 0.910，唯一在 1M 仍维持 0.9+ 表现的模型；
- **LongBench v2 w/ CoT**：56.5（人类 53.7、GPT-4o 51.4、Claude-3.5-Sonnet 46.7）；
- **MTOB 半本书翻译 ChrF Δ**：+45.7（领先）。

更重要的是，它在长上下文上的成本优势是**架构级**的：在长上下文极限下，Lightning Attention 的有效算力消耗只有标准 Softmax 的 1/几千（"百万级长文本只用 1/2700 算力"是 MiniMax 架构负责人钟怡然在公开访谈中给出的估算）。

### 2.3 MiniMax-M2 / M2.5：从基座向 Agent 模型的范式收敛

到了 2025 年 10 月，MiniMax 在 GitHub 开源 **MiniMax-M2**：

- **230B 总参 / 10B 激活**的 MoE 模型；
- 定位是"Mini model built for Max coding & agentic workflows"——刻意把激活参数做小，换取**端到端 agent 闭环（plan → act → verify）的低延迟与高并发**；
- 在 SWE-bench Verified = 69.4、Terminal-Bench = 46.3、ArtifactsBench = 66.8、BrowseComp = 44、τ²-Bench = 77.2、AA Intelligence 综合分 = 61，是同期开源模型中综合能力最强的之一；
- M2 是**交错思考（interleaved thinking）模型**，输出中保留 `<think>...</think>` 段，多轮历史必须原样回传，否则性能显著下滑；
- 后续推出的 **MiniMax-M2.5** 进一步强化全栈编程、工具调用、信息检索与办公场景，定位为"原生 Agent 生产级 LLM"。

注意：M2/M2.5 的命名沿用 M1 系列，但在公开资料中并未强调 M2 是否继续使用 hybrid lightning attention（M2 更接近"小激活、强 agent"路线，技术细节官方信息相对克制）；而 M1 则是**MiniMax-Text-01 架构在推理方向上的直接继承者**。

---

## 3. Lightning Attention 核心技术深度解析

Lightning Attention 是 MiniMax-01 / M1 体系的"地基"。它本质上是**线性注意力（Linear Attention）的 IO-aware 高效实现**，由 Zhen Qin 等人提出（TransNormer / TransNormerLLM 系列工作），在 *Various Lengths, Constant Speed: Efficient Language Modeling with Lightning Attention*（ICML 2024，arXiv:2405.17381）中给出完整算法，并被 MiniMax-01 真正大规模工程化。

### 3.1 标准 Softmax Attention 的瓶颈

经典自注意力为：

$$\mathbf{O} = \text{softmax}\!\big(\mathbf{Q}\mathbf{K}^\top / \sqrt{d}\big)\,\mathbf{V}$$

- 时间复杂度 **O(n²d)**，对长序列不友好；
- KV cache 内存随序列线性增长，4M token 级别的 KV cache 几乎无法承受；
- FlashAttention 系列只是**优化常数与 IO**，理论复杂度仍是 n²。

### 3.2 Linear Attention 的核心思想

线性注意力把 Softmax 拆掉，用核函数把 Attention 拆成 Q · (KᵀV)：

$$\mathbf{O} = \phi(\mathbf{Q})\,\big(\phi(\mathbf{K})^\top \mathbf{V}\big)$$

由于矩阵乘满足结合律，可以**先算 KᵀV ∈ ℝ^{d×d}**，复杂度从 O(n²d) 降到 O(nd²)。在递推形式下还能写成：

$$\mathbf{kv}_t = \mathbf{kv}_{t-1} + \mathbf{k}_t\mathbf{v}_t^\top,\quad \mathbf{o}_t^\top = \mathbf{q}_t^\top \mathbf{kv}_t$$

理论上 KV 状态固定为 d×d，**推理时常数空间、常数复杂度**——天生就是给"百万 token 上下文"准备的。

但是在因果（causal）训练时存在两个老问题：

1. **cumsum 瓶颈**：递推形式必须串行扫描所有时间步，每步累加 KᵀV，无法并行，GPU 利用率极低；
2. **精度差距**：朴素 1+elu 等核近似的线性注意力在语言建模上长期落后于 Softmax。

### 3.3 Lightning Attention 的关键贡献

Lightning Attention 的核心 idea 是**用分块 + intra/inter 双路径计算消除 cumsum**，并把整个算法做成 FlashAttention 风格的 IO-aware kernel：

1. **分块（tiling）**：把 Q/K/V 沿序列维切成 T = n/B 块，每块大小 B×d；
2. **块内（intra-block）走 Left Product**：在块内仍然用 [(QKᵀ) ⊙ M] V 这种"先算 QKᵀ"的左乘范式，因为块小、能并行、能放进 SRAM 共享内存；
3. **块间（inter-block）走 Right Product**：维护一个全局 KV ∈ ℝ^{d×d} 累加器，块结束时把 K_tᵀV_t 累加进 KV，下一块只需要做 Q_{t+1}·KV 即可——这一步不需要逐 token cumsum，而是**逐块累加**；
4. **每块输出 = O_intra + O_inter**，等价于完整 causal 线性注意力，**精确而非近似**；
5. **IO 优化**：所有中间量 Q_t、K_t、V_t、KV 都尽量在 on-chip SRAM 上滚动，最大化避免 HBM 读写，用 Triton 写 kernel；
6. **反向传播同样 tiled**：dQ、dK、dV 和 dKV 都分块计算，不再依赖整序列 cumsum，反向也享有 O(nd²) 复杂度与近常数显存。

理论复杂度（论文 Theorem 3.1）：

$$O(nd^2 + nBd),\quad 取\;B \approx d \;\Rightarrow\; O(nd^2)$$

实测中，与 Vanilla Linear Attention 和 FlashAttention-2 对比：

- **训练吞吐**：FlashAttention-2 的耗时仍随序列长度二次增长，Lightning Attention **保持线性增长**，并且**在不同序列长度下保持常数级训练速度**（这是论文标题"Various Lengths, Constant Speed"的来源）；
- **显存**：Vanilla 线性注意力会因为中间矩阵爆显存，Lightning Attention 显存占用与 FlashAttention-2 同级甚至更低；
- **质量**：在 Wikitext-103 / 大规模 LLM 训练上 perplexity 和下游能力**追平甚至超过 Softmax Transformer**（TransNormerLLM 已在 7B/15B 量级验证，MiniMax-01 把验证规模推到 456B）。

### 3.4 与标准 Attention 的复杂度与表达能力对比

| 维度 | Softmax Attention | FlashAttention-2 | Lightning Attention |
|------|--------------------|-------------------|---------------------|
| 训练时间复杂度 | O(n²d) | O(n²d) | **O(nd²)** |
| 训练显存（理论） | O(n²) → 经 Flash 降到 O(n) | O(n) | **O(n)**，常数极小 |
| 推理 KV 状态 | 随序列线性增长 | 同 Softmax | **固定 d×d** |
| 推理时间（每 token） | O(nd) | O(nd) | **O(d²)** |
| 信息容量 | 显式存储所有 token | 同 Softmax | 隐式压缩进 KV 矩阵 |
| 长程检索 | 强 | 强 | 偏弱（信息被压缩） |
| 长上下文成本 | 二次方爆炸 | 二次方爆炸 | **线性** |

这张对比表也直接解释了 MiniMax-01 为什么不是"纯线性注意力"，而是 **Hybrid 混合注意力**。

### 3.5 Hybrid Lightning + Softmax 混合策略

MiniMax-01 / M1 采用的混合方式是：

> **每 7 层 Lightning Attention 后插入 1 层标准 Softmax Attention**（也就是 8 层为一个 block，1/8 的层是 Softmax）。

设计动机：

1. **质量兜底**：纯线性注意力在 NIAH（Needle-In-A-Haystack）、长程精确检索类任务上的表现不如 Softmax；保留少量 Softmax 层负责"重要 token 的精确召回"；
2. **成本主导 = Lightning**：7/8 的层是 Lightning Attention，长上下文几乎所有计算瓶颈都在 O(nd²) 那一侧，因此推理 4M token 才"可承受"；
3. **效果优于纯 Softmax + MoE**：MiniMax 在内部消融实验中发现 **Hybrid-Lightning + MoE > Softmax + MoE**，论文报告这一点是技术决策的关键证据；
4. **长上下文 + 推理时计算（test-time compute）双友好**：M1 后续做 RL 推理时，长 chain-of-thought 不会让显存爆炸——这是 M1 能做 80K thinking budget 的硬件基础。

### 3.6 上下文窗口扩展：1M 训练 → 4M 推理

MiniMax-Text-01 训练阶段直接把上下文拉到 100 万 token，并通过：

- **RoPE base 提升到 1e7**（远高于 LLaMA 的 1e4），缓解长程角度坍缩；
- 仅对一半头维度施加 RoPE，剩余维度保持 NoPE，配合线性注意力的隐式记忆；
- 长上下文阶段的数据 curriculum 与高质量长文档数据；

实现 **训练 1M、推理外推 4M token**。在 4M Needle-In-A-Haystack 测试中通过率良好，是当时唯一一个把"4M 上下文"**真正交付到开源权重**里的模型。

### 3.7 训练效率收益

线性注意力对 MiniMax 训练栈的收益是结构性的：

- **FLOPs**：长上下文段的注意力 FLOPs 从 n² 降到 n，理论上 1M 上下文相对 4K，节省可达数千倍；
- **激活 / KV 显存**：因为没有 Softmax 中的全 attention map，激活检查点（activation checkpointing）方案大大简化；
- **Pipeline 并行 + Sequence 并行**：MiniMax 为 MoE + Lightning Attention 的组合定制了并行切分与 communication-computation overlap，使得 1M 长度的批量训练在数千卡集群上是可行且经济的；
- **RL 训练亲和度**：M1 的 CISPO RL 训练全程在 512 张 H800 上跑了 3 周，租金 53.47 万美元——这一规模对常规 Softmax 推理模型几乎不可能。

---

## 4. 多模态能力：从 VL 到 Speech / Music / Video

MiniMax 的多模态战略与 OpenAI / Google 类似——**统一基座 + 模态特化**，但更激进地做语音和音乐。

### 4.1 MiniMax-VL-01：把视觉接到 Lightning 基座上

MiniMax-VL-01 在 MiniMax-Text-01 之上**继续训练 5120 亿（512B）视觉-语言 token** 得到，结构是经典的 ViT + MLP Projector + LLM：

- **视觉编码器**：303M 参 ViT，24 层，patch=14，hidden=1024，FFN=4096，16 头 × 64 维；
- **语言主干**：原 MiniMax-Text-01（456B / 45.9B 激活、Hybrid Lightning Attention）；
- 训练做了多阶段：图像描述对齐 → 多任务图文 → 长视觉文档（M-LongDoc）。

**Vision Benchmark 表现**（节选自官方仓库）：

- MMMU = 68.5、MMMU-Pro = 52.7（同档优秀）
- ChartQA = 91.7、DocVQA = 96.4、OCRBench = 865（OCR 与图表是强项）
- M-LongDoc（长视觉文档）= 32.5（明显领先 GPT-4o 41.4 之外的多数同行）
- MathVista = 68.6、In-house Benchmark = 56.6

VL-01 的差异化在于**视觉 + 长上下文**：长 PDF / 多图电子书 / 视频帧序列这种"图文混杂 + 上下文巨大"场景，是 MiniMax-VL-01 相对其他 VL 模型的天然优势。

### 4.2 MiniMax-Speech / Speech-02：可学习 Speaker Encoder + Flow-VAE

语音是 MiniMax 区别于 DeepSeek、Kimi、GLM 等同行最显眼的能力。**MiniMax-Speech**（arXiv:2505.07916，2025-05）的核心系统由三部分构成：

1. **Tokenizer**：文本 BPE + 音频 Encoder-VQ-Decoder（基于 mel-spectrogram，25 token/s，CTC 监督）；
2. **自回归 Transformer**：把文本 token 映射到离散音频 token 序列；
3. **Latent Flow Matching 解码器 + Flow-VAE**：把离散 token 还原成高保真波形。

两项关键创新：

- **Learnable Speaker Encoder（可学习说话人编码器）**：业界主流（CosyVoice2 / Seed-TTS / VALL-E 等）依赖预训练的说话人验证（SV）模型作为冻结的 speaker encoder，而 MiniMax 让 speaker encoder **与 AR Transformer 联合训练**，提取的不是 SV 任务向量，而是真正服务于 TTS 任务的"音色 + 韵律"条件向量。结果是：
  - 严格意义上的 **"intrinsic zero-shot"** 声纹克隆——只需一段无转写参考音频就能克隆说话人；
  - 摆脱对 paired text-audio 的依赖；
  - 因为编码器在所有 32 种语种上联合训练，**跨语种声纹保留能力大幅领先**。

- **Flow-VAE**：传统 VAE 假设潜变量符合标准正态，会形成信息瓶颈；MiniMax 在 VAE encoder 输出后再串一个 Normalizing Flow，把潜分布映射到标准正态后才计算 KL，使得 encoder 实际可以输出**任意正态分布**，潜空间表达力显著提升。Flow Matching 解码器在该潜空间上做生成，比"先生成 mel-spectrogram 再 vocoder"的两段式方案保真度更高。

**评测结果**：

- Seed-TTS-eval：zero-shot WER（test-zh / test-en）= 0.83 / 1.65，**优于 ground truth**；one-shot SIM 0.799 / 0.738，超越 Seed-TTS 与 CosyVoice 2；
- **多语种**：在 Vietnamese、Thai、Cantonese 等"长尾语种"上 WER 远低于 ElevenLabs（ElevenLabs 在 Vietnamese WER 73.4，MiniMax 0.88）；
- **Artificial Arena（2025-05-12）**：MiniMax-Speech（即 Speech-02-HD）以 ELO 第一名超越 OpenAI、ElevenLabs、Google、Microsoft、Amazon。

API 与产品落地：

- 平台侧 `Speech-02-HD`（高保真）/ `Speech-02-Turbo`（更快、更便宜）；
- 海螺 AI 文生音支持最长 10000 字符输入；
- 在海螺 Audio、海螺视频配音、Talkie 数字人对话中全面使用。

### 4.3 MiniMax Music：从 Music 1.5 到 Music 2.5+

- **Music 02**：MoE 架构的 text-to-music 模型，支持歌词/段落控制、风格控制；
- **Music 1.5**（2025）：把生成时长扩到 4 分钟级别；
- **Music 2.0 / 2.5（2026-01-29）**：可控性、真实感再一次上台阶，支持**段落级控制**与人声+乐器协作；
- **Music 2.5+**（2026 上半年）：扩展到纯器乐创作，覆盖古典、电子、ambient、电影配乐等。

整个 Music 路线和 Speech 共用了**音频 token 化 + AR Transformer + Flow Matching 解码器**这一套基础设施。

### 4.4 海螺视频：Video-01 / T2V-01-Director / I2V-01-Director

视频侧，海螺视频（hailuoai.video）依托：

- **Video-01**：text-to-video 通用模型；
- **T2V-01-Director**：文生视频，支持镜头语言（推拉摇移、运镜节奏、构图）显式控制；
- **I2V-01-Director / I2V-01-Live**：图生视频，支持把静态图人物按导演脚本进行表演；

`01-Director` 系列把"专业镜头语言"作为可控属性放到 prompt 里，是海螺视频在 C 端短视频/电影预可视化上的差异化卖点；其海外月访问量已达千万级。

### 4.5 海螺 AI 与 Talkie：双产品矩阵

- **海螺 AI（hailuoai.com / hailuoai.video）**：全面整合文本（基于 abab/MiniMax-01/M-series）、语音（Speech-02）、音乐（Music）、视频（Video-01）、图像；C 端定位"多模态生产力 + 创作力"。
- **Talkie / 星野**：海外/国内的 AI 角色陪伴与创作社区，依托 MiniMax 自研对话与人设模型，海外 Talkie MAU 曾突破 1100 万。
- **MiniMax Open Platform / minimax.io**：B 端 API，提供 ChatCompletion、Realtime、Finetune、T2A、Music、Video 等接口。
- **MiniMax Agent（2025-10 起）**：基于 M2/M2.5 的浏览器/IDE/终端 agent 产品。

商业上，多模态 + 出海是 MiniMax 与国内同行最大的差异。

---

## 5. 推理模型 MiniMax-M1：把 Lightning Attention 扩展到 Test-Time Compute

2025 年 6 月，MiniMax 发布 **MiniMax-M1**（arXiv:2506.13585），副标题直指核心命题：*Scaling Test-Time Compute Efficiently with Lightning Attention*。

### 5.1 模型定位与规模

M1 是**世界上第一个开源权重的大规模 hybrid-attention 推理模型**：

- 基于 MiniMax-Text-01 继续训练，**456B 总参 / 45.9B 激活**；
- 沿用 Lightning Attention + Softmax + MoE 的混合结构；
- **原生支持 1M token 上下文**，是 DeepSeek-R1 的 8 倍；
- 提供两个版本：**M1-40K**（思考预算 40K token，相当于 80K 训练的中间版本）、**M1-80K**（80K 思考预算）。

### 5.2 为什么 Lightning Attention 让 M1 成为可能

推理模型的核心特征是**长 chain-of-thought**：模型可能要消耗 4 万、8 万甚至更多 token 来"思考"。在 Softmax 注意力下，这部分思考 token 的 KV Cache 与 attention 计算线性累加，会让长思考 + 长输入彻底变得不经济。Lightning Attention 把这一段成本压平：

- 长 thinking 的注意力 FLOPs 从 n² 变 n；
- 长 thinking 的 KV state 在 7/8 的层保持常数；
- 长输入（1M）+ 长思考（80K）+ 长输出几乎可以同时存在。

这也是为什么 M1 能在 OpenAI-MRCR-1M（56.2）、LongBench-v2（61.5）等长文本推理任务上明显超越 DeepSeek-R1 / Qwen3-235B。

### 5.3 CISPO：为长 RL 训练定制的策略算法

M1 提出 **CISPO**（Clipped Importance Sampling Policy Optimization），核心想法是：

> **裁剪重要性采样权重，而非裁剪 token 级 update**，绕开传统 PPO/GRPO 中由于策略漂移导致的不稳定。

实测下，CISPO 在与多种主流 RL 变体（GRPO、Dr. GRPO、DAPO 等）的对比中表现更优；与 Hybrid-Attention 结合后：

- **训练总成本**：512 × H800，3 周完成全 RL，租金 ≈ **53.47 万美元**；
- 相对量级模型，是同期最便宜的"达到前沿水平"的 RL 训练之一。

### 5.4 训练数据与任务

M1 的 RL 任务覆盖：

- 数学推理（AIME 2024/2025、MATH-500）
- 代码（LiveCodeBench、FullStackBench）
- **真实软件工程（SWE-bench，沙盒化的真实仓库）**
- 工具使用（TAU-bench airline / retail）
- 长上下文（OpenAI-MRCR 128k/1M、LongBench-v2）

### 5.5 评测结果（节选）

| 任务 | M1-80K | DeepSeek-R1-0528 | Qwen3-235B-A22B | Claude 4 Opus | Gemini 2.5 Pro | OpenAI-o3 |
|------|--------|------------------|------------------|----------------|------------------|-----------|
| AIME 2024 | 86.0 | 91.4 | 85.7 | 76.0 | 92.0 | 91.6 |
| AIME 2025 | 76.9 | 87.5 | 81.5 | 75.5 | 88.0 | 88.9 |
| MATH-500 | 96.8 | 98.0 | 96.2 | 98.2 | 98.8 | 98.1 |
| LiveCodeBench | 65.0 | 73.1 | 65.9 | 56.6 | 77.1 | 75.8 |
| **SWE-bench Verified** | **56.0** | 57.6 | 34.4 | 72.5 | 67.2 | 69.1 |
| **OpenAI-MRCR (128k)** | **73.4** | 51.5 | 27.7 | 48.9 | 76.8 | 56.5 |
| **OpenAI-MRCR (1M)** | **56.2** | -- | -- | -- | 58.8 | -- |
| LongBench-v2 | 61.5 | 52.1 | 50.1 | 55.6 | 65.0 | 58.8 |
| **TAU-bench airline** | **62.0** | 53.5 | 34.7 | 59.6 | 50.0 | 52.0 |

M1 的差异化标签是非常清晰的：

- **长上下文推理**（MRCR 128k/1M、LongBench-v2 全面领先开源同类）；
- **复杂软件工程 / Agent 工具使用**（SWE-bench Verified、TAU-bench airline 在开源端属第一梯队）；
- **训练 / 推理性价比**（hybrid lightning + CISPO）。

短板是数学/AIME 极限分数仍略落后于 DeepSeek-R1-0528 和 Gemini 2.5 Pro，这是 hybrid 注意力对极限符号推理任务上的小折损。

### 5.6 从 M1 到 M2 / M2.5：方向收敛到 Agent

M1 之后，MiniMax 没有继续走"更大基座更长 thinking"的路，而是用 **M2（230B / 10B 激活）** 做了一个重要切换：

- 把激活参数压到 **10B 级别**，让 plan→act→verify 循环延迟和成本骤降；
- 在 SWE-bench、Terminal-Bench、BrowseComp、τ²-Bench 等 Agent 基准全面发力；
- M2 是 **interleaved thinking 模型**，思考与工具调用交错；
- **Artificial Analysis Intelligence 综合分 = 61，开源第一**。

M2 对开发者最大的吸引力是 **"前沿 agent 能力 + 8% Sonnet 成本"** 这条性价比曲线——这与 M1 强调"长上下文推理"是不同的子目标。从 M1 到 M2，MiniMax 完成了一次"从研究展示到产品化 Agent"的姿态切换。**M2.5** 进一步把 M2 推进为生产级 Agent 模型，覆盖编程 / 工具调用 / 信息检索 / 办公场景。

---

## 6. 跨代技术演进分析

把 abab → MiniMax-01 → M1/M2 三代放在同一张表里，可以看到 MiniMax 的演化是一条**"先稀疏化，再线性化，再 RL/Agent 化"** 的清晰路径：

| 维度 | abab6.5（2024-04） | MiniMax-01（2025-01） | MiniMax-M1（2025-06） | MiniMax-M2（2025-10） |
|------|---------------------|------------------------|------------------------|------------------------|
| 总参 / 激活 | ~1T MoE | 456B / 45.9B | 456B / 45.9B（同 01） | **230B / 10B** |
| 注意力 | Softmax | **Hybrid Lightning + Softmax (7:1)** | Hybrid（同 01） | （未公开混合细节） |
| 上下文 | 200k | **训练 1M / 推理 4M** | **原生 1M**（thinking 80K） | 长上下文 + agent |
| 训练目标 | 通用基座 + 对话对齐 | 通用基座 + VL 继续训练 | **大规模 RL + CISPO** | **Agent 端到端 RL** |
| 多模态 | 文本主导 | 文本 + ViT 视觉 | 文本推理 | Coding / Agent / 工具 |
| 主力产品 | 海螺 AI / Talkie | 海螺 AI 长文 / VL | 长文档推理 / 论文级长链 | MiniMax Agent / IDE |
| 战略卡位 | 国内首个 MoE | 开源 4M 上下文 + linear attention 大规模验证 | 第一个开源 hybrid-attention 推理模型 | 开源 agent 综合榜第一 |

**演进逻辑梳理：**

1. **第 1 跳：稠密 → 稀疏（abab5.5 → abab6 → 6.5）。** 把"知识容量"通过 MoE 提一个数量级，单 token 推理成本几乎不变；
2. **第 2 跳：Softmax → Hybrid Lightning（abab6.5 → MiniMax-01）。** 把"上下文长度"再提一个数量级，从 200k 到 4M；
3. **第 3 跳：基座 → 推理模型（MiniMax-Text-01 → M1）。** 在 hybrid 基座上叠加大规模 RL（CISPO），把"思考长度"作为新的扩展维度（test-time compute scaling），而 Lightning Attention 让长思考变得经济；
4. **第 4 跳：推理 → Agent（M1 → M2）。** 把激活参数刻意做小，把"端到端工具循环"的延迟与成本压到极致，从研究模型走向生产级 Agent。

每一步都不是单纯堆参数，而是**换一个新的扩展维度**。Linear / Lightning Attention 在第二跳起就是底盘技术，并且持续帮助第三、第四跳。

---

## 7. 关键创新点（特别突出线性注意力的贡献）

### 7.1 Lightning Attention 的工程突破

在 MiniMax 的体系里，线性注意力第一次从"学术 toy"变成"工业基座"，关键原因有三点：

1. **算法层 cumsum 消除**：tiling + intra/inter 双路径让因果线性注意力既保留了 O(nd²) 的理论复杂度，又获得了 GPU 友好的并行度，**精确等价于完整线性注意力，而非近似**。
2. **系统层 IO-aware kernel**：Triton 实现 + on-chip SRAM 滚动 KV，使前向/反向都接近 FlashAttention 级常数项，但在长序列下时间近似线性增长（"Various Lengths, Constant Speed"）。
3. **混合架构（Hybrid Lightning + Softmax）**：把 7/8 的层换成 Lightning，1/8 的层保留 Softmax 兜底长程检索；并通过 MiniMax-01 在 456B 规模、1M 训练上下文上**实证 Hybrid + MoE 优于纯 Softmax + MoE**——这是当前最具说服力的"线性注意力大规模可行"的实验证据。

### 7.2 Scaling Foundation Models 的新维度

在 GPT-4 / Claude / Gemini 把"更大模型 + 更多数据"作为主旋律时，MiniMax-01 提出了第三个扩展维度：**上下文长度的扩展**——并把它做到了 1M 训练 / 4M 推理，且**开源**。这条路线后续被 M1（test-time compute）、M2（agentic 工具循环）继承和细化。

### 7.3 多模态多产品同基座

MiniMax 是少数把**同一个基座 + 同一套训练栈**真正贯穿到 Text / VL / Speech / Music / Video 五个模态、并且每个模态都有头部产品落地的公司：

- VL-01 直接复用 Text-01 的 hybrid lightning 主干；
- M1 / M2 复用 Text-01 的训练管线 + RL 框架；
- Speech / Music / Video 共用音频/视觉 token 化 + AR Transformer + Flow Matching 解码器范式。

### 7.4 语音（TTS / 声纹克隆）的差异化

可学习 Speaker Encoder + Flow-VAE 的组合让 MiniMax-Speech 在 Artificial Arena 这种**人类盲听偏好**评测上超越 OpenAI / ElevenLabs / Google / Microsoft / Amazon——这在中国大模型公司里是少见的"多模态单点世界第一"。

### 7.5 RL 算法（CISPO）

CISPO 通过裁剪重要性采样权重而非 token 更新，在长链思考的 RL 训练中获得更高的样本效率与稳定性，让 M1 用 53 万美元跑出前沿水平。

### 7.6 商业化 + 出海双轮

abab 时代赌 MoE，MiniMax-01 时代赌 Linear Attention，配合 Talkie / 海螺 AI 在海外市场的强势增长，让 MiniMax 成为"中国 AI 出海"叙事中最有说服力的模板之一（Talkie 海外 MAU 1100 万，Hailuo 视频海外月访问千万级）。

---

## 8. 关键洞察与展望

1. **Lightning Attention 是 MiniMax 体系的护城河**：从 MiniMax-01 到 M1，所有"长上下文 / 长思考 / 高吞吐"优势都来自这一项技术。即使 M2 在公开资料中没有强调注意力混合，整个研发栈对 hybrid attention 的工程经验是其他公司难以短期追上的。
2. **Hybrid 注意力可能是中长期主流**：纯 Softmax 在 1M+ 上下文不可承受，纯线性注意力在精确召回上有损；MiniMax-01 验证 7:1 比例可在 456B 规模收敛得很好，这一组合很可能成为长上下文推理模型的事实标准（参考后续 Tri Dao 等学者的认可）。
3. **测试时计算（Test-time Compute Scaling）会进一步利好线性注意力**：思考链越长，n² 与 nd² 的差距越关键。M1 已经把 thinking budget 推到 80K，未来 200K、500K 的 thinking 几乎只有 hybrid lightning 这类架构能承担。
4. **Agent 路线优先小激活**：M2（230B / 10B 激活）已经验证"小激活 + 强 agent loop"路线，未来 MiniMax 大概率会在 M3 / M3.5 上继续推这条路；与之配合，海螺 / MiniMax Agent 会逐渐覆盖 IDE、Browser、Office、终端的全栈 agent 场景。
5. **多模态的统一仍是开放问题**：Text、VL、Speech、Music、Video 当前共享的是"基础设施层"（tokenizer + AR + Flow Matching），而非"统一模型权重"。如果 MiniMax 后续推出"全模态统一基座"（类似 Qwen-Omni 的路线），其在 Speech / Music / Video 上的领先会进一步放大。

---

## 9. 参考文献

### 核心论文

1. MiniMax et al. *MiniMax-01: Scaling Foundation Models with Lightning Attention*. arXiv:2501.08313, 2025-01-14. <https://arxiv.org/abs/2501.08313>
2. MiniMax et al. *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*. arXiv:2506.13585, 2025-06-16. <https://arxiv.org/abs/2506.13585>
3. Z. Qin, W. Sun, D. Li, X. Shen, W. Sun, Y. Zhong. *Various Lengths, Constant Speed: Efficient Language Modeling with Lightning Attention*. ICML 2024 / arXiv:2405.17381. <https://arxiv.org/abs/2405.17381>
4. MiniMax. *MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder*. arXiv:2505.07916, 2025-05-12. <https://arxiv.org/abs/2505.07916>

### 官方仓库与权重

5. MiniMax-AI. *MiniMax-01* GitHub Repository. <https://github.com/MiniMax-AI/MiniMax-01>
6. MiniMax-AI. *MiniMax-M1* GitHub Repository. <https://github.com/MiniMax-AI/MiniMax-M1>
7. MiniMax-AI. *MiniMax-M2* GitHub Repository. <https://github.com/MiniMax-AI/MiniMax-M2>
8. Hugging Face. *MiniMaxAI/MiniMax-Text-01*. <https://huggingface.co/MiniMaxAI/MiniMax-Text-01>
9. Hugging Face. *MiniMaxAI/MiniMax-VL-01*. <https://huggingface.co/MiniMaxAI/MiniMax-VL-01>
10. Hugging Face. *MiniMaxAI/MiniMax-M1-40k / MiniMax-M1-80k*. <https://huggingface.co/MiniMaxAI>

### 官方博客与发布

11. MiniMax. *MiniMax-01 is Now Open-Source: Scaling Lightning Attention for the Era of Foundation Models*. <https://www.minimax.io/news/minimax-01-series-2>
12. MiniMax. *The General Large Language Model abab6.5 Series*. <https://www.minimax.io/news/abab65-series>
13. MiniMax. *Speech-02-series: The Next Leap in Text-to-Audio and AI Voice Cloning*. <https://www.minimax.io/news/speech-02-series>
14. MiniMax. *MiniMax-M1 Technical Seminar: Deep Dive into RL, Hybrid Attention*. <https://www.minimax.io/news/minimax-m1-technical-seminar-2>
15. MiniMax. *Hailuo AI Advances Cinematic Storytelling with T2V-01-Director and I2V-01-Director*. <https://www.minimax.io/news/01-director>
16. MiniMax 官方中文站. *MiniMax 通用大模型 abab 6.5 系列*. <https://www.minimaxi.com/news/通用大模型abab65系列>
17. MiniMax 开放平台 API 文档. <https://platform.minimax.io/docs>

### 技术解读与第三方分析

18. Neurohive. *MiniMax-01: 4M Context Length Benchmark Leader Powered by Lightning Attention*. <https://neurohive.io/en/state-of-the-art/minimax-01-4m-context-length-benchmark-leader-powered-by-lightning-attention/>
19. QED42. *MiniMax-01: Long-context and Multimodal AI Insights*. <https://www.qed42.com/insights/comprehensive-analysis-of-minimax-01-advancements-in-long-context-processing-and-multimodal-ai>
20. Yacine Mahdid. *How Minimax-01 Achieves 1M Token Context Length with Linear Attention*. <https://www.yacinemahdid.com/p/how-minimax-01-achieves-1m-token>
21. 知乎专栏. *大模型结构基础（八）：MiniMax-01 精读之 Hybrid Lightning Attention*. <https://zhuanlan.zhihu.com/p/19522287467>
22. BAAI Hub. *让百万级长文本只用 1/2700 算力——对话 MiniMax-01 架构负责人钟怡然*. <https://hub.baai.ac.cn/view/44961>
23. 知乎. *[论文速看] MiniMax-M1: CISPO-截断重要性采样权重策略优化*. <https://zhuanlan.zhihu.com/p/1922497761967858562>

### 公司与产品

24. Turing Post. *Yan Junjie and MiniMax: Inside China's AGI Startup Story*. <https://www.turingpost.com/p/minimax>
25. The Wire China. *MiniMax's Moment*. <https://www.thewirechina.com/2025/07/20/minimaxs-moment/>
26. 人民网. *海外月访问量达千万级，MiniMax 海螺 AI 加速视频生成行业发展*.
27. 财联社. *MiniMax 海螺 AI 爆火 海外国产 AI 开启出海掘金之路*. <https://www.cls.cn/detail/1833833>
28. 海螺 AI 视频. <https://hailuoai.video/>
29. Hailuo AI（海螺中文站）. <https://hailuoai.com/>
30. MiniMax Agent 入口. <https://agent.minimax.io/>

### 评测与外部基准

31. Hugging Face Papers 讨论页. *MiniMax-01: Scaling Foundation Models with Lightning Attention*. <https://huggingface.co/papers/2501.08313>
32. Hugging Face Papers. *MiniMax-M1*. <https://huggingface.co/papers/2506.13585>
33. Artificial Analysis Intelligence Benchmark Methodology. <https://artificialanalysis.ai/methodology/intelligence-benchmarking>
34. Artificial Analysis Text-to-Speech Arena. <https://artificialanalysis.ai/text-to-speech/arena>

---

> 本报告依据公开资料、官方技术报告、官方仓库 README 与第三方解读整理；基准数据均引自官方发布版本。MiniMax 系列后续可能持续迭代，最新动态请以 minimax.io / minimaxi.com 与官方 GitHub 为准。
