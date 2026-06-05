# NVIDIA Nemotron 系列研究脉络

> 调研时间：2026-06-06  
> 覆盖范围：NVIDIA Nemotron 系列大语言模型从 2024 年至 2026 年 6 月的完整技术演进

---

## 一、执行摘要

NVIDIA Nemotron 系列是英伟达自研的开源大语言模型家族，其发展脉络清晰展现了从**传统 Dense Transformer** → **合成数据驱动** → **混合 Mamba-Transformer 架构** → **Agentic 多模态全栈模型**的技术演进路径。与多数模型厂商不同，NVIDIA 的核心差异化在于**推理效率优化**与**硬件协同设计**，每一代模型都深度结合 NVIDIA GPU 架构（H100 → Blackwell）和量化技术（FP8 → NVFP4）进行端到端优化。

截至 2026 年 6 月，Nemotron 系列的关键技术里程碑包括：
- **Nemotron-4 340B**（2024-06）首次展示了 98%+ 合成数据驱动的模型对齐 pipeline
- **Llama-Nemotron**（2025-05）首次开源支持动态推理切换（reasoning toggle）的异构推理模型
- **Nemotron-H**（2025-04）引入混合 Mamba-Transformer 架构，推理速度提升最高 3 倍
- **Nemotron 3**（2025-12~2026-06）形成完整的三层 Agentic 家族：Nano（30B/3.5B）→ Super（120B/12B）→ Ultra（550B/55B），采用 Mamba-2 + Attention MoE 混合架构，配合 Latent-MoE、MTP、NVFP4 训练，成为 Agentic AI 的高效基座
- **Nemotron Coalition**（2026-03）联合八大 AI 实验室共建开源前沿模型，Nemotron 4 正在开发中

---

## 二、模型演进时间线

| 时间 | 模型 / 论文 | 核心定位 | 关键创新 |
|------|------------|---------|---------|
| 2024-02 | **Nemotron-4 15B** (arXiv:2402.16819) | 中型 Dense 基座模型 | 预训练数据配方与评估基准 |
| 2024-06 | **Nemotron-4 340B** (arXiv:2406.11704) | 大型合成数据生成器 | 98%+ 合成数据对齐、Reward Model、开放 Pipeline |
| 2025-03 | **Llama 3.1 Nemotron Nano 8B** | 端侧高效推理模型 | 基于 Llama 3.1 蒸馏 + 优化 |
| 2025-03 | **Llama 3.3 Nemotron Super 49B** | 中型推理模型 | 基于 Llama 3.3 改进 |
| 2025-03 | **Llama 3.1 Nemotron 70B Instruct** | 指令遵循模型 | 长上下文、工具使用 |
| 2025-04 | **Llama 3.1 Nemotron Ultra 253B** | 大型推理模型 | 企业级复杂推理 |
| 2025-04 | **Nemotron-H** (arXiv:2504.03624) | 混合架构基座模型 | Mamba + Transformer 混合、FP8 训练、MiniPuzzle 压缩 |
| 2025-05 | **Llama-Nemotron** (arXiv:2505.00949) | 异构推理模型家族 | 动态推理切换、RL 训练、开源 Post-Training Dataset |
| 2025-08 | **Nemotron Nano 2** (arXiv:2508.14444) | 高效推理小模型 | Mamba-2 + Attention、6.3× 吞吐提升、20T tokens 预训练 |
| 2025-08 | **Nemotron Nano 9B v2** | Nano 迭代版 | 知识截止 2024-09，多语言与推理增强 |
| 2025-12 | **Nemotron 3 Nano** (arXiv:2512.20856) | Agentic 轻量基座 | Mamba-2 + Attention MoE、Latent-MoE、MTP、NVFP4、1M 上下文 |
| 2026-03 | **Nemotron 3 Super** (120B/12B) | 中型 Agentic 模型 | SWE-Bench Verified 60.47%、Intelligence Index 36、1M 上下文 |
| 2026-03 | **Nemotron 3 子系列** | 垂直领域扩展 | Speech、RAG、Safety、VoiceChat、Omni |
| 2026-03 | **Nemotron Coalition** | 开放联盟 | 联合 Black Forest Labs、Cursor、LangChain、Mistral 等八大实验室 |
| 2026-06 | **Nemotron 3 Ultra** (550B/55B) | 旗舰 Agentic 模型 | Intelligence Index 48（美国开源第一）、300+ tok/s、262K/1M 上下文 |

---

## 三、分代技术详解

### 3.1 第一代：Nemotron-4 系列（2024）—— 合成数据驱动的对齐范式

#### Nemotron-4 15B（2024-02）
- **论文**：*Nemotron-4 15B Technical Report* (arXiv:2402.16819)
- **定位**：中型 Dense Transformer 基座模型，探索预训练数据配方
- **规模**：15B 参数，训练数据细节公开

#### Nemotron-4 340B（2024-06）
- **论文**：*Nemotron-4 340B Technical Report* (arXiv:2406.11704)
- **定位**：大型开源模型家族，核心亮点是**合成数据生成（Synthetic Data Generation, SDG）**
- **模型家族**：
  - **Nemotron-4-340B-Base**：340B 参数，9T tokens 预训练
  - **Nemotron-4-340B-Instruct**：指令遵循版本
  - **Nemotron-4-340B-Reward**：奖励模型，RewardBench 上超越 GPT-4o 和 Gemini 1.5 Pro
- **核心创新**：
  - **98%+ 合成数据对齐**：整个对齐阶段超过 98% 的数据为合成生成，大幅降低人工标注成本
  - **合成数据 Pipeline**：开源完整的 prompt 生成 → response 生成 → 质量过滤 → 偏好排序 pipeline
  - **Iterative Weak-to-Strong**：多轮迭代自增强，用中间模型生成更高质量数据训练下一代
  - **LLM-as-Judge / Reward-Model-as-Judge**：系统比较了两种自动评估策略，最终 Reward Model 在 Chat-Hard 类别上显著优于 LLM（0.87 vs 0.54）
- **许可证**：NVIDIA Open Model License（允许商业应用）
- **影响**：为后续 Nemotron 系列奠定了"合成数据 + 自迭代"的技术基因，也成为社区重要的合成数据生成基座

---

### 3.2 第二代：Llama-Nemotron 系列（2025-03~05）—— 基于 Llama 的高效推理改进

2025 年初，NVIDIA 基于 Meta Llama 3.1/3.3 基座，通过**神经架构搜索（NAS）**、**知识蒸馏**、**持续预训练**和**大规模强化学习**，推出了一系列改进模型：

| 模型 | 发布时间 | 基座 | 规模 | 核心能力 |
|------|---------|------|------|---------|
| Llama-3.1-Nemotron-Nano-8B-v1 | 2025-03-18 | Llama 3.1 8B | 8B | 端侧高效推理 |
| Llama-3.1-Nemotron-70B-Instruct | 2025-03-18 | Llama 3.1 70B | 70B | 指令遵循、长上下文 |
| Llama-3.3-Nemotron-Super-49B-v1 | 2025-03-18 | Llama 3.3 70B | 49B | 中型推理（剪枝后） |
| Llama-3.1-Nemotron-Ultra-253B-v1 | 2025-04-07 | Llama 3.1 405B | 253B | 大型推理（蒸馏后） |

#### Llama-Nemotron（2025-05，arXiv:2505.00949）
- **论文**：*Llama-Nemotron: Efficient Reasoning Models*
- **定位**：首个开源**异构推理模型家族**，支持动态推理模式切换
- **模型规模**：
  - **LN-Nano**：8B（基于 Llama 3.1 8B）
  - **LN-Super**：49B（基于 Llama 3.3 70B 剪枝）
  - **LN-Ultra**：253B（基于 Llama 3.1 405B 蒸馏）
- **核心创新**：
  - **动态推理切换（Dynamic Reasoning Toggle）**：首个开源模型支持通过轻量级 system prompt（"detailed thinking on/off"）在标准聊天模式和深度推理模式间动态切换，无需切换模型或架构
  - **五阶段训练流程**：
    1. NAS + FFN Fusion 优化推理效率
    2. 知识蒸馏 + 持续预训练恢复精度
    3. SFT 混合标准指令数据与 DeepSeek-R1 等强教师的推理轨迹
    4. 大规模 RL 训练（数学/STEM 数据集），LN-Ultra 在 GPQA-D 上超越教师模型
    5. 短对齐阶段优化指令遵循和人类偏好
  - **FP8 生成优化**：为大规模 RL 训练开发自定义框架，支持 FP8 精度生成
  - **开源资源**：完整 Post-Training Dataset、NeMo / NeMo-Aligner / Megatron-LM 训练代码
- **性能**：LN-Ultra 在 2025-04 的 Artificial Analysis Intelligence Index 上被评为"最聪明的开源模型"，在多项推理和非推理基准上领先

---

### 3.3 第三代：Nemotron-H / Nano 2（2025-04~08）—— 混合 Mamba-Transformer 架构

#### Nemotron-H（2025-04，arXiv:2504.03624）
- **论文**：*Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models*
- **定位**：探索 Mamba 线性复杂度与 Transformer 精确注意力融合的混合架构
- **模型规模**：
  - **Nemotron-H-8B**：8B 参数
  - **Nemotron-H-56B-Base**：56B 参数（FP8 训练）
  - **Nemotron-H-47B-Base**：56B 经 MiniPuzzle 压缩后，推理速度再提升 20%
- **架构设计**：
  - 用 **Mamba 层**替换大部分自注意力层（约 8% 层保留注意力）
  - 保留少量注意力层用于精确长程依赖建模
  - 无位置编码，使用 RMSNorm，独立嵌入/输出层权重
- **核心创新**：
  - **MiniPuzzle 压缩技术**：结合剪枝与知识蒸馏，将 56B 模型压缩至 47B，精度几乎无损，推理更快
  - **FP8 训练方案**：证明 FP8 训练可达到与 BF16 相当的精度，降低训练成本
  - **视觉语言扩展**：Nemotron-H-8B/56B-VLM，采用 NVLM-D 架构，用于 Cosmos-Reason1 等物理 AI 推理
- **性能**：与同规模 Transformer（Qwen-2.5、Llama-3.1）相比，精度相当或更优，**推理速度最高提升 3 倍**

#### Nemotron Nano 2（2025-08，arXiv:2508.14444）
- **论文**：*NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model*
- **定位**：Nemotron-H 的推理专用增强版，面向端侧和高效推理场景
- **基座**：Nemotron-Nano-12B-v2-Base（62 层：6 注意力 + 28 FFN + 28 Mamba-2）
- **核心改进**：
  - **20 万亿 tokens 预训练**：使用 Warmup-Stable-Decay 学习率调度，FP8 精度
  - **128K 长上下文扩展**：持续预训练阶段扩展上下文长度，不损害其他基准
  - **数据集升级**：Nemotron-CC-v2（新增 8 组 CommonCrawl 快照、15 语言合成 QA）、Nemotron-CC-Math-v1（133B 数学 tokens）
  - **剪枝与对齐**：从 12B Base 剪枝得到 9B 模型，再经对齐训练
- **性能**：与 Qwen3-8B 相比，在 1k input / 8k output 或 8k input / 16k output 等生成重载场景下，**吞吐量提升 3×–6.3×**，精度相当或更优

---

### 3.4 第四代：Nemotron 3 家族（2025-12~2026-06）—— Agentic AI 的全栈基座

Nemotron 3 是 NVIDIA 面向 Agentic AI 时代推出的全新模型家族，采用自研的 **Mamba-2 + Attention MoE** 混合架构，从 2025 年底的 Nano 开始，到 2026 年 6 月 Ultra 的发布，形成完整的三层产品梯队。

#### 3.4.1 Nemotron 3 Nano（2025-12）
- **论文**：*NVIDIA Nemotron 3: Efficient and Open Intelligence* (arXiv:2512.20856)
- **定位**：轻量级 Agentic 基座，面向边缘设备和本地部署
- **规格**：30B 总参数 / 3.5B 激活参数（A3B）
- **核心架构**：
  - **混合 Mamba-2 + Attention MoE**：Mamba-2 层处理长序列，Attention MoE 处理精确推理
  - **Latent-MoE**：将 routed expert 计算压缩到低维 latent 空间，减少通信负载和内存占用
  - **Multi-Token Prediction (MTP)**：预测多个未来 token，天然支持 speculative decoding 加速
  - **GQA + 2 KV Heads**：极少 KV 头，降低 KV Cache 内存
  - **1M Token 原生上下文**：支持百万级 token 长文档、代码库、多轮对话记忆
- **训练创新**：
  - **NVFP4 量化训练**：在 Blackwell 架构上实现 4-bit 浮点训练，Nano 模型相对 BF16 损失差距 <1%
  - **多环境强化学习（NeMo Gym）**：在开源 Agent 环境中训练工具使用、规划、验证、多步问题解决能力
  - **开放训练 Pipeline**：数据集、权重、详细 recipe 全部开源
- **性能**：Nemotron 3 Nano 的吞吐量是 Nemotron 2 Nano 的 **4 倍**

#### 3.4.2 Nemotron 3 Super（2026-03-11）
- **定位**：中型 Agentic 模型，面向企业级多 Agent 协作场景
- **规格**：120B 总参数 / 12B 激活参数（A12B），MoE 架构
- **上下文窗口**：**1M tokens**（BF16 原生支持）
- **关键性能**：
  - **SWE-Bench Verified**：60.47%
  - **Intelligence Index**：36 分
  - **知识截止**：2025-06
- **架构延续**：与 Nano 相同的 Mamba-2 + Attention MoE + Latent-MoE + MTP 架构
- **部署**：支持 DeepInfra（$0.10/M input, $0.50/M output）、NVIDIA NIM、OpenRouter、Hugging Face

#### 3.4.3 Nemotron 3 Ultra（2026-06-01 宣布，06-04 发布）
- **发布场合**：GTC Taipei 2026 / Computex 2026 主题演讲，黄仁勋亲自宣布
- **定位**：旗舰 Agentic 模型，面向最复杂的企业级推理和长运行 Agent 工作流
- **规格**：**550B 总参数 / 55B 激活参数**（A55B），MoE 架构
- **上下文窗口**：
  - BF16 精度：**262,144 tokens**
  - NVFP4 量化（Blackwell）：**1M tokens**
- **训练数据**：20 万亿 tokens
- **关键性能**：
  - **Intelligence Index**：**48 分**——截至 2026-06 美国开源模型最高分
  - **推理速度**：300+ tokens/second（DeepInfra 实测）
  - **成本优势**：比同级开源模型低约 30% 的推理成本
  - **吞吐效率**：比同类前沿模型快 3-6 倍
- **对比定位**：
  - 相比 Kimi K2.6（Intelligence Index 54）：Ultra 在绝对智能分上略低，但在速度（300+ vs 50-100 tok/s）和成本效益上显著领先
  - 相比 Claude Opus 4.8 / GPT-5.5：Ultra 在 SWE-bench 等编码基准上仍有差距，但优势在于**开源权重 + 可自托管 + NVIDIA 硬件生态**
- **部署渠道**：Hugging Face、ModelScope、OpenRouter、build.nvidia.com（NIM 微服务）
- **硬件要求**：BF16 权重需要多 GPU 集群（A100/H100 级别），NVFP4 在 Blackwell 上更可行；消费级硬件无法本地运行

#### 3.4.4 Nemotron 3 子系列（2026-03）
在 GTC 2026 上，NVIDIA 同步发布了 Nemotron 3 的多个垂直领域子系列：

| 子系列 | 定位 | 关键特性 |
|--------|------|---------|
| **Nemotron 3 Omni** | 多模态理解 | 融合音频、视觉与语言理解，跨模态无缝协同 |
| **Nemotron 3 VoiceChat** | 实时语音对话 | 集成 ASR + LLM + TTS，全双工语音 Agent |
| **Nemotron Speech** | 开源 ASR | 声称比同级竞品快 10 倍，WER 相当 |
| **Nemotron RAG** | 企业检索 | 多模态 embedding + reranker VLM |
| **Nemotron Safety** | 内容安全 | Llama-Nemotron Content Safety + PII 检测器，覆盖 23 类安全类别、9 种语言 |

---

## 四、关键技术对比

| 技术维度 | Nemotron-4 340B | Llama-Nemotron | Nemotron-H | Nemotron 3 Nano | Nemotron 3 Super | Nemotron 3 Ultra |
|---------|-----------------|----------------|------------|-----------------|------------------|------------------|
| **架构** | Dense Transformer | Dense Transformer (Llama-based) | Hybrid Mamba-Transformer | Hybrid Mamba-2 + Attention MoE | Hybrid Mamba-2 + Attention MoE | Hybrid Mamba-2 + Attention MoE |
| **核心优化目标** | 合成数据质量 | 推理效率 + 动态推理 | 推理速度 | Agentic 边缘效率 | Agentic 企业协作 | Agentic 旗舰推理 |
| **总参数** | 340B | 8B~253B | 8B~56B | 30B | 120B | **550B** |
| **激活参数** | 340B | 8B~253B | 8B~47B | 3.5B | 12B | **55B** |
| **MoE** | ❌ | ❌ | ❌ | ✅ (Latent-MoE) | ✅ (Latent-MoE) | ✅ (Latent-MoE) |
| **Mamba** | ❌ | ❌ | ✅ (Mamba-1) | ✅ (Mamba-2) | ✅ (Mamba-2) | ✅ (Mamba-2) |
| **上下文长度** | 标准 | 128K | 标准 | 1M | **1M** | 262K / 1M (NVFP4) |
| **量化训练** | FP8 推理 | FP8 生成 | FP8 训练 | NVFP4 训练 | NVFP4 训练 | NVFP4 训练 |
| **动态推理切换** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **开源程度** | 模型 + Pipeline | 模型 + 数据集 + 代码 | 模型 + 代码 | 模型 + 数据 + 环境 + 代码 | 模型 + 权重 + recipe | 模型 + 权重 + recipe |
| **硬件绑定** | DGX H100 | 通用 GPU | 通用 GPU | Blackwell 优化 | Blackwell 优化 | **Blackwell 优化** |
| **Intelligence Index** | — | — | — | — | 36 | **48** |

---

## 五、生态与影响

### 5.1 开源贡献
NVIDIA Nemotron 系列在开源社区的影响力体现在：
- **合成数据 Pipeline**：Nemotron-4 340B 的开源 SDG pipeline 成为社区重要工具
- **Post-Training Dataset**：Llama-Nemotron 开源了完整的后训练数据集
- **训练框架**：NeMo、NeMo-Aligner、Megatron-LM、NeMo Gym、NeMo RL 等工具链
- **推理优化**：TensorRT-LLM、NIM 微服务对 Nemotron 架构的原生支持
- **Nemotron-Pre-Training-Dataset-v1**：6.6 万亿 tokens，包含 Nemotron-CC-v2、Nemotron-CC-Math-v1 等

### 5.2 商业采用
Nemotron 3 发布时宣布的早期合作伙伴：
- **Accenture**、**CrowdStrike**、**Oracle Cloud**、**Palantir**、**Perplexity**、**ServiceNow**、**Siemens**、**Zoom**
- **CrowdStrike**：将 Nemotron 模型用于漏洞修复 AI Agent
- **Palantir**：整合至 AI Forward Deployed Engineer 平台，实现任务自主执行
- 覆盖制造、网络安全、软件开发、企业自动化等领域

### 5.3 Nemotron Coalition（2026-03-16，GTC 2026）
NVIDIA 在 GTC 2026 宣布牵头成立**开放前沿联盟**，旨在联合开发下一代开源前沿模型：
- **创始成员**（8 家）：Black Forest Labs、Cursor、LangChain、Mistral AI、Perplexity、Reflection AI、Sarvam、Thinking Machines Lab
- **合作模式**：成员贡献数据、评估、研究领域专长；NVIDIA 提供 DGX Cloud 训练算力
- **首个成果**：**Nemotron 4** 正在开发中，由 NVIDIA 与 Mistral AI 联合开发基座模型，将支撑 Nemotron 4 家族
- **战略意义**：NVIDIA 从"芯片制造商"向"全栈 AI 基础设施平台"转型的关键一步

### 5.4 下载量与社区影响
- Nemotron 3 系列在 2026 年 4 月前累计下载量突破 **5,000 万次**
- Nemotron 3 Ultra 发布首日即登上 Hugging Face、ModelScope、OpenRouter 热榜

---

## 六、研究脉络总结

NVIDIA Nemotron 系列的技术演进可以概括为四条主线：

1. **数据飞轮**：从 Nemotron-4 340B 的 98% 合成数据对齐，到 Llama-Nemotron 的开源 Post-Training Dataset，再到 Nemotron 3 的 20T tokens 预训练数据与开放 Agent 训练环境——NVIDIA 持续推动"数据生成 → 模型训练 → 数据增强"的自迭代飞轮

2. **架构效率**：从 Dense Transformer → 混合 Mamba-Transformer（Nemotron-H）→ Mamba-2 + Attention MoE（Nemotron 3），每一步都围绕**降低推理成本**和**提升长上下文处理能力**展开。Latent-MoE 将通信压缩到低维空间，MTP 提供原生 speculative decoding，GQA+2 KV Heads 极致压缩 KV Cache

3. **硬件协同**：从 H100 上的 FP8 推理 → Blackwell 上的 NVFP4 训练，Nemotron 系列始终与 NVIDIA 最新 GPU 架构深度绑定，形成"模型-软件-硬件"全栈优化壁垒。NVFP4 训练使 550B 模型在 4-bit 精度下仍保持 <1% 损失差距

4. **Agentic 生态**：从单一聊天模型 → 动态推理切换（Llama-Nemotron）→ 多环境 RL 训练（NeMo Gym）→ 全双工语音 Agent（VoiceChat）+ 多模态理解（Omni）+ 企业检索（RAG）+ 内容安全（Safety），Nemotron 正在构建完整的 Agentic AI 基础设施栈

---

## 七、参考来源

| # | 论文 / 报告 | 时间 | 链接 |
|---|------------|------|------|
| 1 | Nemotron-4 15B Technical Report | 2024-02 | https://arxiv.org/abs/2402.16819 |
| 2 | Nemotron-4 340B Technical Report | 2024-06 | https://arxiv.org/abs/2406.11704 |
| 3 | Nemotron-H: Hybrid Mamba-Transformer | 2025-04 | https://arxiv.org/abs/2504.03624 |
| 4 | Llama-Nemotron: Efficient Reasoning Models | 2025-05 | https://arxiv.org/abs/2505.00949 |
| 5 | Nemotron Nano 2 | 2025-08 | https://arxiv.org/abs/2508.14444 |
| 6 | Nemotron 3 Nano | 2025-12 | https://arxiv.org/abs/2512.20856 |
| 7 | NVIDIA Nemotron 3 Ultra Review (BuildFastWithAI) | 2026-06 | https://www.buildfastwithai.com/blogs/nvidia-nemotron-3-ultra-review-2026 |
| 8 | NVIDIA GTC Taipei 2026 解析 (AIPostHub) | 2026-06 | https://www.aiposthub.com/nvidia-gtc-taipei-2026-enterprise-ai-agent-nemotron-3-ultra/ |
| 9 | NVIDIA Nemotron 3 Ultra Review (AIToolsRecap) | 2026-06 | https://aitoolsrecap.com/Blog/nvidia-nemotron-3-ultra-550b-computex-2026 |
| 10 | Nemotron Coalition 宣布 (GetAIBook) | 2026-03 | https://getaibook.com/news/nvidia-launches-nemotron-coalition-at-gtc-2026/ |
| 11 | NVIDIA Nemotron 官方页面 | — | https://research.nvidia.com/labs/nemotron/ |
| 12 | Hugging Face Nemotron 模型集合 | — | https://huggingface.co/nvidia |

---

*报告由 AI Agent 自动调研生成，基于公开论文、技术博客、官方发布信息和第三方评测整理。数据截至 2026-06-06。*
