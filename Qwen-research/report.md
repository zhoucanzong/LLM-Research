# Qwen系列模型深度调研报告

> 调研日期：2025年6月（2026年6月更新）  
> 覆盖范围：Qwen系列从初代（2023年8月）到Qwen3.7-Plus（2026年6月）的完整技术演进  
> 写作原则：以**演进路径**为骨架，每个方向作为一条连续时间线，2026年新进展融入对应章节，而非独立罗列

---

## 1. 概述与发展时间线

### 1.1 系列定位

Qwen（通义千问）是阿里巴巴云推出的全栈大模型系列，自2023年8月首次发布以来已发展为覆盖**纯文本理解与生成、视觉-语言多模态、音频理解与生成、全模态统一感知、长程自主Agent，以及代码/数学/推理等专业方向**的完整模型族群。回顾近三年的演进，其核心设计哲学可归纳为四条：

1. **开放生态优先**：绝大多数模型采用Apache 2.0许可证开源，全面适配HuggingFace、vLLM、Ollama、SGLang、llama.cpp等主流框架；2026年1月HuggingFace累计下载量突破7亿
2. **基座驱动多模态**：以纯文本基座模型为核心，各多模态方向通过编码器+适配器接入基座LLM，基座升级带动所有下游方向"免费"获得能力提升
3. **规模化数据工程**：预训练数据从初代3T tokens持续倍增至Qwen3的36T tokens，每代约翻倍
4. **从对话到Agent**：2025年Qwen3引入Thinking模式，2026年Qwen3.5/3.6/3.7全面转向"Agent原生"基座定位

### 1.2 完整发展时间线（2023–2026）

```
2023年 ── 起步与多模态铺底
├── 08月  Qwen-7B/14B           首个中英双语大语言模型基座（arXiv:2309.16609）
├── 08月  Qwen-VL                首个视觉语言模型（ViT-bigG + Cross-Attn + Qwen-7B）
├── 11月  Qwen-Audio             首个通用音频理解模型（Whisper + Qwen-7B）
└── 12月  Qwen-72B                首个百亿级旗舰

2024年 ── 工程化、规模化与架构质变
├── 02月  Qwen1.5 系列           统一32K上下文 + 首个MoE + HF transformers原生集成
├── 03月  CodeQwen1.5            首个代码专用模型
├── 06月  Qwen2 系列             GQA全面应用 + 7T数据 + DCA/YARN支持128K
├── 06月  Qwen2-VL               Naive Dynamic Resolution + M-RoPE
├── 07月  Qwen2-Audio            DPO对齐 + 双模式音频交互
├── 08月  Qwen2-Math             数学专用模型
├── 09月  Qwen2.5 系列           18T数据 + 128K原生 + 100万SFT + 多阶段RL
├── 09月  Qwen2.5-Coder          5.5T代码token专精训练
├── 09月  Qwen2.5-Math           CoT+PoT+TIR三范式数学推理
├── 10月  Qwen2.5-VL             Window/Full混合注意力 + 绝对时间编码 + 从头训练ViT
└── 11月  QwQ-32B-Preview        首个推理专用模型预览版

2025年 ── 推理革命与全模态统一
├── 01月  Qwen2.5-VL 正式版      72B旗舰对标GPT-4o，新增DPO后训练阶段
├── 03月  QwQ-32B                正式版推理模型（32B≈DeepSeek-R1 671B性能）
├── 03月  Qwen2.5-Omni           首个端到端全模态模型（Thinker-Talker + TMRoPE）
├── 04月  Qwen3 LLM 系列         Thinking/Non-Thinking双模式 + 36T数据 + GRPO + QK-Norm
├── 05月  Qwen3-VL / Qwen3-Omni  多模态最新迭代（Interleaved-MRoPE，原生256K）
├── 07月  Qwen3-Coder 480B-A35B  7.5T代码数据训练，SWE-Bench Verified SOTA
├── 09月  Qwen3-Max              万亿参数旗舰（API-only）
├── 09月  Qwen3-Next-80B-A3B     **混合注意力新架构**（Gated DeltaNet + Gated Attn）
├── 10月  Qwen3-VL-8B-Thinking   视觉推理增强变体
└── 11月  Qwen3-VL 技术报告       arXiv:2511.21631

2026年 ── Agent原生与超长上下文时代
├── 01月  HuggingFace累计下载    突破7亿
├── 02月  Qwen3.5 (397B-A17B)    256专家MoE + 201语言 + 262K原生上下文 + 19x吞吐
├── 02月  Qwen3.5-Plus            托管API版本
├── 02月  Qwen3-VL-Flash          轻量级视觉理解上线百炼
├── 04月  Qwen3.5-Omni            全模态升级（arXiv:2604.15804），215基准SOTA
├── 05月  Qwen3.6 系列            27B/72B Dense + 35B-A3B MoE + Plus/Flash/Max-Preview
├── 05月  Qwen3.7-Max             闭源旗舰Agent模型（1M上下文，35小时自主执行）
├── 05月  Qwen3.5-LiveTranslate   实时同声传译（60语言，2.8秒延迟，声音克隆）
├── 05月  Qwen3-Coder-Next        本地编码Agent（80B-A3B混合注意力）
└── 06月  Qwen3.7-Plus            多模态Agent基座（Vision Arena全球前五）
```

### 1.3 技术代际划分

从架构与训练范式的演进视角，Qwen系列可清晰划分为四个代际：

| 阶段 | 时间跨度 | 代表基座 | 核心技术标志 | 战略定位 |
|------|----------|----------|--------------|----------|
| **奠基期** | 2023.08–2024.02 | Qwen / Qwen1.5 | BBPE 151K大词表、QKV Bias、MHA→GQA过渡、首个MoE | 中英双语对话模型 |
| **规模化期** | 2024.06–2024.12 | Qwen2 / Qwen2.5 | GQA全面统一、7T→18T数据、DCA/YARN 128K、DPO/多阶段RL | 通用大模型旗舰 |
| **推理增强期** | 2025.04–2025.09 | Qwen3 / QwQ | Thinking模式、GRPO、QK-Norm、36T数据、全模态统一、万亿参数Qwen3-Max | 推理增强通用基座 |
| **Agent原生期** | 2025.09– | Qwen3-Next / 3.5 / 3.6 / 3.7 | Hybrid Attention（Gated DeltaNet）、256专家MoE、262K→1M上下文、Native MTP、长程自主执行 | "Foundation Models for the Agent Era" |

每一代的过渡均由**一项关键架构变革**驱动：奠基期→规模化期由GQA统一驱动，规模化期→推理增强期由Thinking模式与GRPO驱动，推理增强期→Agent原生期由Hybrid Attention与1M级超长上下文驱动。

---

## 2. 纯文本基座模型演进（Qwen → Qwen3.7）

纯文本基座是Qwen系列的核心，所有多模态与专业方向均派生自此。下面以**一条连续时间线**呈现从Qwen初代（MHA + 8K上下文）到Qwen3.7-Max（Hybrid Attention + 1M上下文）的完整演进。

### 2.1 跨代架构对比表（覆盖全部代次）

| 组件 | Qwen (2023.08) | Qwen1.5 (2024.02) | Qwen2 (2024.06) | Qwen2.5 (2024.09) | Qwen3 (2025.04) | Qwen3-Next (2025.09) | Qwen3.5 (2026.02) | Qwen3.6 (2026.05) | Qwen3.7-Max (2026.05) |
|------|----------------|-------------------|-----------------|-------------------|-----------------|----------------------|--------------------|--------------------|------------------------|
| 注意力 | MHA | MHA(110B:GQA) | **GQA全部** | GQA | GQA + **QK-Norm** | **Hybrid: Gated DeltaNet + Gated Attn** | Hybrid (GDN) | Hybrid (GDN) | Hybrid (GDN) |
| 位置编码 | RoPE(base=10K) | RoPE | RoPE(base=1M) | RoPE | RoPE + ABF | RoPE + GDN | RoPE + GDN | RoPE + GDN | RoPE + GDN |
| FFN激活 | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| 归一化 | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| QKV Bias | **有** | 有 | 有 | 有 | **移除** | 移除 | 移除 | 移除 | 移除 |
| 词表 | 151,936 | 151,936 | 151,646 | 151,646 | 151,669 | ~151,669 | ~151,669 | ~151,669 | ~151,669 |
| MoE配置 | — | A2.7B(14.3B) | 64+8共享 / 8激活 | Turbo/Plus(API) | **128专家/8激活/无共享** | 高稀疏MoE | **256专家+1共享/8激活** | 35B-A3B | (闭源未公布) |
| Token预测 | 单token | 单token | 单token | 单token | 单token | **Native MTP** | Native MTP | Native MTP | Native MTP |
| 最大上下文 | 8K→32K | 32K | 32K→128K | 128K(1M:Turbo) | 128K(4x推理) | 超长序列优化 | **262K原生** | 128K(YaRN) | **1M** |
| 预训练数据 | ~3T | ~3T | 7T (0.5B:12T) | **18T** | **36T** | 36T+ | 36T+ | 36T+ | 36T+ |
| 语言数 | 中英为主 | 12+ | ~30 | ~30 | **119** | 119+ | **201** | 201 | 201 |

**演进读法**：从左到右扫描每一行，可以看到清晰的"演进事件"——
- **注意力机制**经历三次大跃迁：MHA→GQA（Qwen2）→QK-Norm（Qwen3）→Hybrid Attention（Qwen3-Next起）
- **MoE架构**经历四次重塑：无→Qwen1.5首试→Qwen2细粒度+共享专家→Qwen3移除共享专家+global-batch balance→Qwen3.5重新引入1个共享专家+256路由专家
- **上下文长度**呈指数级增长：8K→32K→128K→262K→1M，五个数量级跨越
- **数据规模**几乎每代翻倍：3T→3T→7T→18T→36T

### 2.2 各代关键突破（演进叙事）

#### Qwen初代（2023.08）— 奠定基础架构

模型规模覆盖1.8B/7B/14B/72B四档，全部采用MHA + RoPE(base=10K) + SwiGLU + Pre-RMSNorm的标准Decoder-only架构。三项关键设计影响整个系列：

- **151,936词表BBPE分词器**：相比LLaMA 32K词表，中文编码效率提升约3-4倍，奠定了Qwen多语言能力的底层基础
- **QKV Bias**：在Q/K/V投影中均引入bias项，作为系列架构标志延续四代直至Qwen3移除
- **上下文扩展三件套**：NTK-aware interpolation + LogN-Scaling + Window Attention，将基础8K扩展至32K（仅72B原生32K，其余通过推理扩展）

后训练采用SFT + RLHF（PPO）的标准范式。

#### Qwen1.5（2024.02）— 工程化与生态化

不发布单独技术论文，以官方博客形式发布。主要做了两件事：**生态对齐**和**MoE首试**。

- **首个MoE模型** Qwen1.5-MoE-A2.7B：14.3B总参/2.7B激活，引入**细粒度专家 + 共享专家**机制
- **统一32K上下文**：所有规模模型标准化32K支持
- **DPO引入**：首次将Direct Preference Optimization纳入对齐管线，形成SFT + DPO + PPO三段式
- **HF transformers原生集成**（≥4.37.0），无需`trust_remote_code`，vLLM/SGLang/llama.cpp/Ollama全面支持
- **110B首试GQA**：在大尺寸模型中引入GQA，为Qwen2全面切换铺路

#### Qwen2（2024.06）— 架构质变

技术报告arXiv:2407.10671。这是Qwen系列**最具决定性的一次架构升级**：

- **GQA全面应用**：所有规模统一切换至GQA。例如Qwen2-7B采用Q头28/KV头4，Qwen2-72B采用Q头64/KV头8，KV Cache内存降低为1/4–1/8
- **RoPE基频提升**：从10,000→1,000,000，配合**DCA（Dual Chunk Attention）+ YARN**支持128K推理上下文
- **数据规模跃迁**：7T tokens（0.5B小模型用12T），多语言扩展至约30种
- **MoE成熟化**：Qwen2-57B-A14B采用64路由专家 + 8共享专家 + 8激活专家
- **Online DPO + Online Merging Optimizer**：在线RLHF新范式，相比纯离线DPO进一步逼近PPO性能

#### Qwen2.5（2024.09）— 数据与对齐双重飞跃

技术报告arXiv:2412.15115。架构与Qwen2基本一致，但训练规模再次大幅升级：

- **18T tokens预训练**（相比Qwen2再翻2.5倍），常识/专家知识/推理能力数据比重显著加强
- **新增模型规模**：3B/14B/32B填补空白
- **128K原生上下文**：7B以上模型原生支持，Turbo/Plus通过专门技术支持百万token
- **100万+样本SFT**（相比Qwen2的50万翻倍）
- **多阶段强化学习**：长文本生成、结构化数据分析、指令遵循各阶段独立优化
- **作为专业模型基座**：直接派生Qwen2.5-Coder/Math、QwQ等专业方向

#### Qwen3（2025.04）— 推理范式革命

技术报告arXiv:2505.09388。这是Qwen系列**最大的一次训练方法论升级**：

- **Thinking/Non-Thinking双模式统一**：单一模型通过`/think`和`/no_think`切换深度推理与快速响应，支持精细的Thinking Budget预算控制
- **QK-Norm替代QKV Bias**：移除延续四代的Bias设计，以归一化方案提升大规模训练稳定性
- **GRPO强化学习**：基于rule-based rewards的Group Relative Policy Optimization，3,995组query-verifier对，Qwen3-235B-A22B在170步RL训练后AIME'24从70.1提升至85.1
- **36T tokens + 119种语言**：使用Qwen2.5-VL从PDF提取文本、Qwen2.5-Math/Coder生成合成数据
- **MoE架构简化**：移除共享专家，128路由专家/8激活 + Global-batch load balancing loss
- **后训练四阶段流水线**：Long-CoT Cold Start → Reasoning RL（GRPO）→ Thinking Mode Fusion → General RL
- **Strong-to-Weak蒸馏**：仅需四阶段方法1/10的GPU时间训练小模型，同时提升Pass@1与Pass@64

预训练分三阶段：S1通用30T+ tokens / S2推理~5T tokens / S3长上下文（4K→32K）。

#### Qwen3-Next-80B-A3B（2025.09）— 混合注意力新架构

这是从Qwen3到Qwen3.5/3.6/3.7的**架构跨越点**，引入了三项关键变革：

- **Hybrid Attention**：用 **Gated DeltaNet（线性注意力变体）+ Gated Attention** 替代标准全注意力。Gated DeltaNet在长序列推理下开销大幅低于Full Attention，与Gated Attention混合后兼顾表达能力与效率
- **Native MTP（Multi-Token Prediction）**：在原生预训练中内置多token预测目标
- **更高稀疏度MoE**：80B总参数仅3B激活（激活比1/27，远超Qwen3的1/10）
- 开源Apache 2.0，可本地部署

Qwen3-Next虽然规模仅80B，但其架构成为**后续所有Qwen3.5/3.6/3.7的技术底座**。

#### Qwen3.5（2026.02）— MoE效率革命

2026年2月16日（农历除夕）发布的首个重大版本升级。架构延续Qwen3-Next的Hybrid Attention，但MoE规模与上下文实现新突破：

- **397B总参/17B激活**：256路由专家 + 8激活 + **重新引入1个共享专家**（Qwen3移除后再次回归）
- **262K原生上下文**：相比Qwen3的128K再翻一倍
- **201种语言/方言**：相比Qwen3的119种大幅扩展（约+70%）
- **吞吐量飞跃**：32K上下文下比Qwen3-Max快**8.6倍**，256K下快**19倍**
- **Apache 2.0完全开源**
- 战略定位"Towards Native Multimodal Agents"

Qwen3.5还推出了Qwen3.5-9B等小尺寸变体，在多个基准上击败更大规模的模型。

#### Qwen3.6 系列（2026.05）— 开源新基线

在Qwen3.5基础上继续迭代，提供完整的开源模型矩阵：

| 模型 | 类型 | 关键参数 |
|------|------|----------|
| Qwen3.6-27B | Dense | 单A100可部署，工作马模型 |
| Qwen3.6-72B | Dense | MMLU 88.5% / HumanEval 92.1% |
| Qwen3.6-35B-A3B | MoE | 3B激活，128K上下文（YaRN），R1-Zero RLHF对齐 |
| Qwen3.6-Plus | API | 生产级托管版本 |
| Qwen3.6-Flash | API | 低延迟优化 |
| Qwen3.6-Max-Preview | API | 实验性前沿能力 |

工程优化：**AITemplate内核融合**，推理吞吐相比Qwen2.5提升2倍；R1-Zero风格RLHF对齐替代传统SFT+DPO。

#### Qwen3.7-Max（2026.05.19）— Agent Frontier旗舰

闭源旗舰，专为长程自主Agent设计：

- **1M tokens上下文** + 65,536 tokens最大输出
- **35小时连续自主运行**记录：在86小时RL训练中自主标记1,618个奖励黑客（reward hacking）案例
- **跨框架泛化**：在Claude Code / OpenClaw / Qwen Code等不同Agent框架中通用
- **API兼容**：同时支持OpenAI spec和Anthropic spec
- 持续优化案例：432次kernel评估、1,158次工具调用

关键基准（vs Claude Opus 4.6）：

| 基准 | Qwen3.7-Max | Claude Opus 4.6 |
|------|-------------|-----------------|
| Terminal Bench 2.0 | **69.7** | 65.4 |
| SWE-Verified | 80.4 | 80.8 |
| GPQA Diamond | **92.4** | 91.3 |
| IMOAnswerBench | **90.0** | 75.3 |
| LiveCodeBench | **91.6** | 88.8 |
| HMMT 2026 Feb | **97.1** | 96.2 |

### 2.3 训练规模与对齐方法演进

| 代次 | 预训练数据 | 语言覆盖 | SFT样本 | 对齐方法 |
|------|-----------|---------|---------|----------|
| Qwen | ~3T tokens | 中英为主 | — | SFT + PPO |
| Qwen1.5 | ~3T tokens | 12+ | — | SFT + DPO + PPO |
| Qwen2 | 7T (0.5B:12T) | ~30 | 50万+ | SFT + DPO + Online DPO + Online Merging |
| Qwen2.5 | **18T** | ~30 | **100万+** | SFT + 多阶段RL |
| Qwen3 | **36T** | **119** | — | CoT SFT + **GRPO** + Mode Fusion + General RL |
| Qwen3.5/3.6 | 36T+ | **201** | — | GRPO + **R1-Zero RLHF** |
| Qwen3.7-Max | 36T+（含Agent数据） | 201 | — | 长程Agent RL（自监控reward hack） |

### 2.4 MoE架构演进（一条独立的演进线）

MoE是Qwen系列演进最频繁的子模块之一：

| 代次 | 模型 | 总/激活 | 路由专家 | 激活 | 共享专家 | 均衡策略 |
|------|------|---------|---------|------|---------|----------|
| Qwen1.5 | MoE-A2.7B | 14.3B/2.7B | — | — | 有 | — |
| Qwen2 | 57B-A14B | 57B/14B | 64 | 8 | **8个共享** | 辅助loss |
| Qwen3 | 30B-A3B | 30B/3B | **128** | 8 | **移除** | Global-batch balance |
| Qwen3 | 235B-A22B | 235B/22B | **128** | 8 | **移除** | Global-batch balance |
| Qwen3-Next | 80B-A3B | 80B/3B | 高稀疏度 | — | (新设计) | — |
| Qwen3.5 | 397B-A17B | 397B/17B | **256** | 8 | **回归1个** | — |
| Qwen3.6 | 35B-A3B | 35B/3B | (公开未详) | — | — | — |

**演进规律**：专家数量呈指数增长（无→8→64→128→256），共享专家经历"引入→移除→再次引入1个"的反复，激活参数比从Qwen2的1/4降至Qwen3.5的1/23（约5x稀疏化）。

---

## 3. 视觉语言模型演进（Qwen-VL → Qwen3.7-Plus）

VL方向经历了从"借用OpenCLIP外部组件"到"端到端自研架构"再到"统一Agent基座"的三段式演进。

### 3.1 VL架构演进脉络

```
Qwen-VL (2023)              Qwen2-VL (2024.06)          Qwen2.5-VL (2024.10)        Qwen3-VL (2025.05/11)       Qwen3.7-Plus (2026.06)
┌──────────────┐          ┌─────────────────┐          ┌──────────────────┐        ┌──────────────────┐         ┌──────────────────┐
│ OpenCLIP      │          │ 自定义ViT 675M   │          │ 从头训练ViT       │        │ Qwen3-VL ViT      │         │ 统一多模态编码器   │
│ ViT-bigG      │          │ + 2D-RoPE        │          │ + Window/Full Attn│        │ + Interleaved-MRoPE│         │ (图像/视频/屏幕/网页)│
│ 固定448×448   │          │ + 任意分辨率      │          │ + RMSNorm+SwiGLU │        │ + 256K交错原生     │         │                   │
└──────┬────────┘          └──────┬───────────┘          └──────┬───────────┘        └──────┬───────────┘         └──────┬───────────┘
       ▼                          ▼                             ▼                             ▼                             ▼
┌──────────────┐          ┌─────────────────┐          ┌──────────────────┐        ┌──────────────────┐         ┌──────────────────┐
│ Cross-Attn    │          │ MLP (2×2→1)     │          │ MLP-based Merger │        │ MLP Merger       │         │ MLP Merger        │
│ 256 queries   │          │ 动态token        │          │ 动态token         │        │ 动态token         │         │ 动态token          │
│ 固定256 tokens │          │                  │          │                   │        │                   │         │                    │
└──────┬────────┘          └──────┬───────────┘          └──────┬───────────┘        └──────┬───────────┘         └──────┬───────────┘
       ▼                          ▼                             ▼                             ▼                             ▼
┌──────────────┐          ┌─────────────────┐          ┌──────────────────┐        ┌──────────────────┐         ┌──────────────────┐
│ Qwen-7B LLM   │          │ Qwen2 LLM       │          │ Qwen2.5 LLM      │        │ Qwen3 LLM        │         │ Qwen3.7 LLM       │
│ + 标准RoPE    │          │ + M-RoPE        │          │ + 增强M-RoPE     │        │ + Interleaved-MRoPE│        │ + Hybrid Attention │
│               │          │                  │          │ + 绝对时间编码    │        │ + Thinking模式    │         │ + Visual Agent能力 │
└───────────────┘          └─────────────────┘          └──────────────────┘        └──────────────────┘         └──────────────────┘
```

### 3.2 跨代关键对比表

| 维度 | Qwen-VL (2023) | Qwen2-VL (2024) | Qwen2.5-VL (2025.01) | Qwen3-VL (2025.11) | Qwen3.7-Plus (2026.06) |
|------|----------------|-----------------|----------------------|--------------------|------------------------|
| 视觉编码器 | OpenCLIP ViT-bigG | 自定义ViT 675M (DFN初始化) | 从头训练ViT (Window+Full混合) | 增强ViT | 统一多模态编码器 |
| 分辨率策略 | 固定448×448 | Naive Dynamic Resolution | 原生动态分辨率(增强) | 原生动态 | 原生动态 |
| Token数量 | 固定256 | 动态(H/28×W/28) | 动态(原生分辨率) | 动态 | 动态 |
| ViT归一化/激活 | LayerNorm + GELU | LayerNorm + GELU | **RMSNorm + SwiGLU**（与LLM统一）| RMSNorm + SwiGLU | RMSNorm + SwiGLU |
| 融合方式 | Cross-Attn Resampler | MLP(2×2→1) | MLP-based Merger | MLP Merger | MLP Merger |
| LLM位置编码 | 标准RoPE | **M-RoPE** | 增强M-RoPE + 绝对时间 | **Interleaved-MRoPE** | Interleaved-MRoPE |
| 计算复杂度 | O(n²) | O(n²) | **~O(n)（近线性）** | ~O(n) | ~O(n) |
| 视频支持 | 不支持 | 固定2FPS, 16384 token上限 | 动态FPS, 小时级长视频, 秒级时间定位 | 256K交错原生 | 视频/屏幕Agent |
| 坐标表示 | — | 归一化坐标 | **绝对像素坐标** | 绝对像素 | 绝对像素 |
| Agent能力 | 基础Bbox定位 | UI操作/机器人/导航 | Computer Use + Phone Use | GUI Agent | **Multimodal Agent闭环** |
| Thinking模式 | — | — | — | **Qwen3-VL-8B-Thinking** | 内置 |
| 模型尺寸 | 仅7B | 2B/7B/72B | 3B/7B/72B | 2B/8B/72B + Flash | API only |

### 3.3 三大设计哲学转变（贯穿VL全代次）

1. **从固定到动态**：分辨率固定→动态、token数固定→动态、帧率无→固定→动态、坐标无→归一化→绝对像素
2. **从复杂到简洁**：Cross-Attention Resampler → MLP投影、绝对位置编码 → RoPE相对编码、独立ViT架构 → 与LLM统一为RMSNorm+SwiGLU
3. **从借用到自研**：OpenCLIP预训练 → DFN初始化 → 完全从头训练；标准位置编码 → M-RoPE → Interleaved-MRoPE原创方案

### 3.4 关键技术创新（按代次首发）

#### Qwen2-VL首创：Naive Dynamic Resolution

传统VLM要么将图像缩放至固定分辨率（信息损失），要么切割为固定子图（伪影）。Qwen2-VL方案极为简洁：

- **移除ViT中的绝对位置嵌入**，改用2D-RoPE使ViT具备处理任意尺寸输入的能力
- 视觉token数与图像尺寸严格成正比：`(H/14)×(W/14)`
- MLP层将相邻2×2 token合并，最终token数为`(H/28)×(W/28)`
- 保留原始纵横比，无需padding或裁剪

#### Qwen2-VL首创：M-RoPE — 多模态位置编码统一方案

将RoPE的频率维度均分为三个独立子空间：

| 模态 | Temporal ID | Height ID | Width ID |
|------|------------|-----------|----------|
| 文本 | position | position | position |
| 图像 | 常量 | 行位置 | 列位置 |
| 视频 | 帧索引→绝对时间 | 帧内行位置 | 帧内列位置 |

#### Qwen2.5-VL进化：Window Attention + 绝对时间编码

- **Window Attention混合设计**：ViT大部分层使用窗口≤8×8的Window Attention，仅4层使用Full Attention，复杂度从O(n²)降至接近O(n)
- **绝对时间编码**：M-RoPE的temporal ID与真实秒数对齐（不再用帧索引），配合动态FPS训练，支持秒级事件定位（`mm:ss.ff`格式）
- **绝对坐标**：Bbox使用图像原始像素坐标（非归一化）
- **ViT架构与LLM统一**：LayerNorm→RMSNorm，GELU→SwiGLU
- 后训练新增**DPO阶段**

#### Qwen3-VL（2025.05/11）：增强Interleaved-MRoPE + 256K原生交错

技术报告arXiv:2511.21631。三大支柱：

1. **纯文本理解显著增强**：在多场景超越同级别纯文本模型
2. **多模态全面升级**：图像/视频/文档SOTA
3. **Agent/GUI原生设计**：面向GUI Agent工作流

架构升级：增强的Interleaved-MRoPE提供更强时空建模 + 原生256K交错图文长序列 + Qwen3-VL-8B-Thinking变体支持视觉推理增强。

#### Qwen3-VL-Flash（2026.02）

轻量级视觉理解模型，2026年2月上线阿里云百炼，面向低成本生产部署。

#### Qwen3.7-Plus（2026.06）：多模态Agent基座

Vision Arena全球前五、中国第一。与Qwen3-VL的"纯视觉理解"不同，Qwen3.7-Plus将视觉与语言**统一为一体化Agent基座**：

| 能力 | 说明 |
|------|------|
| Multimodal Agent | 统一处理图像、视频、屏幕、网页和文本，在GUI/CLI/工具环境中完成任务 |
| Visual Agent | 视觉理解 + 代码解释器 + 搜索增强，解决视觉谜题与复杂推理 |
| Visual Coding | 从图像/视频生成SVG、网页、交互式前端 |
| GUI Agent | 移动端/桌面端界面理解、控件定位、任务规划与多步操作 |
| Real-world Perception | 文档图表、OCR、视频、驾驶场景理解 |

实现"看、想、写、做、验"的端到端智能体工作流，仅通过阿里云百炼API提供。

### 3.5 VL方向的演进总结

VL方向的演进，本质上是**视觉编码器从"附加组件"逐步内化为"原生模态"**的过程：
- Qwen-VL：视觉是被压缩到256 token的"外挂"
- Qwen2-VL：视觉获得动态尺寸表达权 + M-RoPE统一位置
- Qwen2.5-VL：ViT与LLM架构同构（RMSNorm+SwiGLU）+ 真实时间感
- Qwen3-VL：视觉token与文本token交错原生处理（256K）
- Qwen3.7-Plus：视觉成为Agent决策的等位输入，与代码、工具、屏幕一并进入闭环

---

## 4. 多模态与Omni模型演进

Omni方向是Qwen过去三年最具突破性的一条线：从单模态音频理解→双模式音频交互→端到端全模态→实时同声传译。

### 4.1 Omni方向跨代对比

| 维度 | Qwen-Audio (2023.11) | Qwen2-Audio (2024.07) | Qwen2.5-Omni (2025.03) | Qwen3-Omni (2025.05) | Qwen3.5-Omni (2026.04) | Qwen3.5-LiveTranslate (2026.05) |
|------|----------------------|----------------------|------------------------|-----------------------|------------------------|--------------------------------|
| 模态范围 | 仅音频理解 | 仅音频理解 | **文本/图像/音频/视频统一** | 文本/图像/音频/视频统一 | 全模态+SOTA | 实时翻译专精 |
| 音频编码器 | Whisper-large-v2 | Whisper-large-v3 + 池化层 | 块式处理(Block-wise) | 块式处理 | (升级版) | (升级版) |
| LLM基座 | Qwen-7B | Qwen-7B | Qwen2.5 (Thinker) | Qwen3 | Qwen3.5 | Qwen3.5 |
| 训练范式 | 层次化标签多任务 | **自然语言提示** | 端到端联合训练 | 联合训练 | 联合训练 | 联合训练 |
| 对齐方法 | 仅SFT | SFT + **DPO** | SFT+DPO | GRPO | GRPO | GRPO |
| 总参数量 | ~7.7B | 8.2B | 7B (Thinker+Talker) | 数十亿 | 数百亿 | 优化版 |
| 生成能力 | 仅理解 | 仅理解 | **流式语音生成(Talker)** | 流式生成 | 可控音视频字幕 | **声音克隆+实时翻译** |
| 位置编码 | 标准 | 标准 | **TMRoPE** (时间对齐) | TMRoPE+ | TMRoPE+ | TMRoPE+ |
| 关键基准 | LibriSpeech 2.0% WER | AIR-Bench全面超Gemini-1.5-pro | Omni-Bench SOTA | 4模态同时SOTA | **215基准SOTA**（超Gemini 3.1 Pro） | 60语言/2.8秒延迟 |

### 4.2 各代关键突破（演进叙事）

#### Qwen-Audio（2023.11）— 统一多任务音频理解

论文arXiv:2311.07919。**首个**通过大规模预训练统一覆盖语音、自然音、音乐三大类音频理解任务的模型。

- **架构**：Whisper-large-v2编码器 + Qwen-7B LLM，音频特征作为前缀token直接注入LLM
- **核心创新：层次化标签多任务训练**——通过Transcription Tag / Audio Language Tag / Task Tag / Text Language Tag / Timestamps Tag / Output Instruction六层标签解决多任务联合训练干扰
- **SRWT（词级时间戳识别）**：训练模型在识别语音的同时预测每个词的时间戳，开始时间在词前、结束时间在词后
- **训练数据**：14万+小时，30+种音频任务（ASR 30K小时 + S2TT 3.7K + 音乐描述 25K等）
- **训练策略**：预训练冻结LLM→微调冻结编码器
- **关键性能**：LibriSpeech test-clean 2.0% WER（前最佳2.1%），CoVoST2 en-zh BLEU 41.5（前最佳33.1）

#### Qwen2-Audio（2024.07）— 自然语言交互升级

论文arXiv:2407.10759，总参数8.2B。三大改进：

| 维度 | 与Qwen-Audio对比 |
|------|-----------------|
| 音频编码器 | Whisper-large-v2 → **Whisper-large-v3 + 池化层(stride=2)**，每帧约40ms原始音频 |
| 训练范式 | 层次化标签 → **自然语言提示**（更好的泛化与指令跟随）|
| 对齐技术 | 仅SFT → **SFT + DPO**（人类标注好坏对，提升事实性） |
| 交互模式 | 单一 → **语音聊天 + 音频分析双模式**（无需系统提示切换） |

**关键性能**：LibriSpeech test-clean 1.6% WER（vs Qwen-Audio 2.0%），AIR-Bench Speech 7.18 / Sound 6.99 / Music 6.79 / Mixed 6.77，**全面超越Gemini-1.5-pro**。

#### Qwen2.5-Omni（2025.03）— 端到端全模态

论文arXiv:2503.20215，参数7B。Qwen系列**里程碑式**的突破，首次在单一模型中统一文本/图像/音频/视频的感知与生成。

**Thinker-Talker双模块架构**（灵感源自人脑"思考"与"表达"分离）：

| 模块 | 角色 | 功能 | 实现 |
|------|------|------|------|
| **Thinker** | "大脑" | 接收多模态输入，生成文本响应 + 高维隐藏表示 | 大型LLM（基于Qwen2.5）|
| **Talker** | "嘴巴" | 接收Thinker隐藏表示，生成语音 | 双轨自回归Transformer解码器（灵感源自Mini-Omni）|

设计优势：联合训练避免文本与语音模态干扰；语音生成直接基于Thinker高维语义表示，确保语义一致性；真正的端到端架构而非级联TTS。

**TMRoPE（Time-aligned Multimodal RoPE）**：从M-RoPE进化而来，核心改进是**将音频和视频按时间顺序交错排列**，确保不同模态的时间戳在位置编码中精确对齐。

**流式处理设计**：
- 音频/视觉编码器：Block-wise块式处理 → 实时流式输入
- 音频解码：Sliding-window DiT → 减少首包延迟
- 整体架构：流式生成支持实时对话

**性能定位**：视觉理解≈Qwen2.5-VL；音频理解>Qwen2-Audio；语音指令跟随≈纯文本输入（MMLU、GSM8K性能相当）；Omni-Bench综合SOTA。

#### Qwen3-Omni（2025.05）— 在Thinker-Talker基础上的五大升级

在Qwen2.5-Omni的Thinker-Talker架构基础上进行五大关键升级，成为**首个在文本、图像、音频、视频上同时保持SOTA**的单一模型。

#### Qwen3.5-Omni（2026.04）— 全模态升级

技术报告arXiv:2604.15804。三大新能力：

- **可控的音视频字幕生成**：精细控制字幕风格与内容
- **百亿级参数规模**，音频-视觉理解和生成均达SOTA
- 旗舰版Qwen3.5-Omni-Plus在**215个音频/音视频基准中达到SOTA，超越Gemini 3.1 Pro**
- 视觉编码器采用Qwen3.5的统一编码器处理图像和视频

#### Qwen3.5-LiveTranslate-Flash（2026.05.19）— 实时同声传译

基于Qwen3.5-Omni构建的实时同声传译模型：

| 特征 | 详情 |
|------|------|
| 支持语言 | 60种（29种语音 + 31种文本） |
| 端到端延迟 | **2.8秒** |
| 声音克隆 | 自动复制说话者声音特征 |
| 视觉增强 | 支持多模态实时翻译 |
| 应用场景 | 会议、直播、视频通话 |

### 4.3 Omni方向的演进主线

```
Qwen-Audio (2023)        Qwen2-Audio (2024)       Qwen2.5-Omni (2025)        Qwen3.5-Omni (2026)         Qwen3.5-LiveTranslate (2026)
─────────────────        ──────────────────       ───────────────────        ───────────────────         ────────────────────────────
分模态独立处理      ──→   分模态独立处理      ──→  全模态端到端统一       ──→  全模态SOTA统一         ──→   实时翻译专用
层次化标签训练      ──→   自然语言提示训练    ──→  联合训练              ──→  联合训练               ──→   声音克隆+实时
Whisper-v2          ──→   Whisper-v3 + 池化   ──→  块式处理编码器        ──→  统一多模态编码器        ──→   优化版编码器
仅理解              ──→   理解+DPO对齐        ──→  理解+流式语音生成     ──→  可控字幕生成           ──→   实时多模翻译
单一7B               ──→   8.2B               ──→  7B (Thinker+Talker)   ──→  数百亿参数             ──→   优化版
```

主线清晰：**模态范围扩张 → 训练范式简化 → 生成能力增强 → 实时性突破**。

---

## 5. 专业方向模型演进

### 5.1 Coder方向：CodeQwen1.5 → Qwen2.5-Coder → Qwen3-Coder → Qwen3-Coder-Next

| 模型 | 时间 | 规模 | 关键特征 |
|------|------|------|----------|
| **CodeQwen1.5** | 2024.03 | 7B | Qwen系列首个代码专用模型 |
| **Qwen2.5-Coder** | 2024.09 | 0.5B/1.5B/3B/7B/14B/32B | Qwen2.5基座 + **5.5T代码token续训**；保留通用语言+数学能力；HumanEval/MBPP等10+基准SOTA；7B击败更大通用模型 |
| **Qwen3-Coder 480B-A35B** | 2025.07 | 480B总/35B激活 | **7.5T tokens训练**（70%代码），1M上下文（YaRN），SWE-Bench Verified SOTA，匹配Claude Sonnet 4 |
| **Qwen3-Coder-Next** | 2026.05 | 80B总/3B激活 | **混合注意力架构**（Gated DeltaNet），256K上下文，专为本地编码Agent设计，达到Sonnet 4.5级编码性能 |

**演进规律**：从"通用基座续训"（CodeQwen1.5/Qwen2.5-Coder）→ "代码专用旗舰"（480B-A35B）→ "本地Agent化"（Coder-Next 3B激活）。Coder-Next是**首次将Qwen3-Next架构应用到代码方向**，使得本地编码Agent成为可能（80B激活仅3B，单卡可推理）。

设计原则一以贯之：**保留通用语言和数学能力的同时专精代码**，避免"代码模型不会做数学"的常见陷阱。

### 5.2 Math/推理方向：Qwen2-Math → Qwen2.5-Math → QwQ → Qwen3 Thinking

| 模型 | 时间 | 规模 | 路线 |
|------|------|------|------|
| **Qwen2-Math** | 2024.08 | — | 数学专用首作 |
| **Qwen2.5-Math** | 2024.09 | 1.5B/7B/72B | **CoT + PoT + TIR三范式**；Qwen2-Math生成合成数据自我改进；72B超越GPT-4o |
| **QwQ-32B-Preview** | 2024.11 | 32B | 首个推理专用模型预览版 |
| **QwQ-32B** | 2025.03 | 32B | 正式版，**规模化RL训练**（非SFT/RLHF），32B≈DeepSeek-R1（671B/37B激活）性能 |
| **Qwen3 Thinking模式** | 2025.04 | 全系列 | 推理能力**融入主基座**；GRPO + rule-based rewards；235B-A22B AIME'24从70.1→85.1 |
| **Qwen3-VL-8B-Thinking** | 2025.10 | 8B | 推理范式扩展到VL |
| **Qwen3.7-Max** | 2026.05 | (闭源) | IMOAnswerBench 90.0、HMMT 2026 Feb 97.1，**长程Agent推理新维度** |

**演进规律**：从"数学专用模型"（Qwen2-Math/2.5-Math）→ "推理专用模型"（QwQ-32B）→ "推理融入主基座"（Qwen3 Thinking模式）→ "Agent级长程推理"（Qwen3.7-Max的35小时自主执行）。

QwQ的核心贡献是验证了**规模化RL（无需大规模SFT）即可获得深度推理能力**，这条路线被Qwen3的GRPO范式继承并放大。Qwen2.5-Math的三范式CoT+PoT+TIR则成为后续所有推理模型的基本能力套件。

### 5.3 专业模型与基座的关系（基座驱动专业化）

```
Qwen2.5 (18T通用预训练)
    │
    ├── + 5.5T代码token ────→ Qwen2.5-Coder
    ├── + 数学合成数据 ──────→ Qwen2.5-Math
    ├── + 规模化RL ─────────→ QwQ-32B
    ├── + ViT编码器 ────────→ Qwen2.5-VL
    └── + 音频编码器 ───────→ Qwen2-Audio → Qwen2.5-Omni

Qwen3 (36T预训练 + Thinking)
    │
    ├── + 7.5T代码token ────→ Qwen3-Coder 480B-A35B
    ├── + Interleaved-MRoPE → Qwen3-VL
    ├── + Thinker-Talker ───→ Qwen3-Omni
    └── + 蒸馏 ─────────────→ 小模型矩阵

Qwen3-Next架构（Hybrid Attention + Native MTP）
    │
    ├── 普惠扩展 ──────────→ Qwen3.5 (397B-A17B)
    ├── 普惠扩展 ──────────→ Qwen3.6 系列
    ├── 普惠扩展 ──────────→ Qwen3.7-Max（1M上下文Agent）
    ├── 普惠扩展 ──────────→ Qwen3-Coder-Next（本地Agent）
    └── 普惠扩展 ──────────→ Qwen3.5-Omni → LiveTranslate
```

每次基座升级，所有下游方向均获得"免费"的能力提升。Qwen3的Strong-to-Weak蒸馏与Qwen3-Next的Hybrid Attention成为**两次最重要的"普惠技术注入"**——前者使小模型训练效率提升10倍，后者使所有后续方向获得超长上下文能力。

---

## 6. 跨代技术演进深度分析

本章纵向梳理几个核心架构维度从Qwen初代到Qwen3.7的完整演化轨迹，提炼设计哲学的转变。

### 6.1 注意力机制：MHA → GQA → QK-Norm → Hybrid Attention

| 时间节点 | 决策 | 动机 |
|----------|------|------|
| 2023.08 (Qwen) | MHA（每个Q头独立KV） | 标准设计，最大表达能力 |
| 2024.02 (Qwen1.5-110B) | GQA首试 | 大模型推理效率需求 |
| 2024.06 (Qwen2) | **GQA全面统一** | KV Cache内存降低为1/4–1/8 |
| 2025.04 (Qwen3) | GQA + **QK-Norm**（移除QKV Bias） | 归一化稳定大规模训练 |
| 2025.09 (Qwen3-Next起) | **Hybrid: Gated DeltaNet + Gated Attention** | 超长序列推理效率 |

注意力机制是Qwen系列**变革频率最高**的架构维度。Qwen3-Next的Hybrid Attention是关键拐点：通过线性注意力变体（Gated DeltaNet）替代大部分Full Attention层，使1M级上下文成为可能。

### 6.2 位置编码：RoPE → M-RoPE → TMRoPE → Interleaved-MRoPE

| 时间节点 | 文本模型 | 视觉模型 | 全模态模型 |
|----------|---------|---------|-----------|
| 2023 | RoPE (base=10K) | ViT绝对位置 + LLM标准RoPE | — |
| 2024 | RoPE (base=1M) + DCA + YARN | **2D-RoPE (ViT) + M-RoPE (LLM)** | — |
| 2025 (前) | RoPE + ABF | M-RoPE + 绝对时间编码 | **TMRoPE**（时间对齐） |
| 2025 (后) | RoPE + ABF | **Interleaved-MRoPE** | TMRoPE+ |

位置编码是Qwen系列最具创新性的技术线索：从标准RoPE出发→M-RoPE统一文本/图像/视频位置→TMRoPE实现音频-视频时间对齐→Interleaved-MRoPE支持原生交错长序列。

### 6.3 MoE演进：无 → 8 → 64 → 128 → 256专家

```
Qwen          无MoE
  │
Qwen1.5       MoE-A2.7B (首个MoE) — 细粒度专家+共享专家
  │
Qwen2         57B-A14B (64+8共享/8激活) — 辅助loss均衡
  │
Qwen3         128专家/8激活/无共享 — Global-batch balance
  │
Qwen3-Next    高稀疏度MoE (3B/80B = 1/27激活)
  │
Qwen3.5       256专家+1共享/8激活 (17B/397B)
  │
Qwen3.6       35B-A3B + 多档API版本
```

**演进规律**：专家数量呈指数级（无→8→64→128→256），共享专家经历"引入→移除→以更精简形式回归（Qwen3.5的1个共享专家）"，激活比从1/4稀疏化至1/23。

### 6.4 训练范式演进：SFT+RLHF → DPO → Online DPO → GRPO → R1-Zero RLHF

```
Qwen      SFT + PPO（人类标注reward）
  │
Qwen1.5   SFT + DPO + PPO（离线偏好优化引入）
  │
Qwen2     SFT + DPO + Online DPO + Online Merging（在线学习）
  │
Qwen2.5   SFT + 多阶段RL（长文本/结构化/指令各阶段独立优化）
  │
Qwen3     CoT SFT + GRPO + Mode Fusion + General RL（rule-based验证）
  │
Qwen3.5/3.6  GRPO + R1-Zero风格RLHF
  │
Qwen3.7-Max  长程Agent RL（自监控reward hacking，86小时训练标记1,618个hack案例）
```

**核心趋势**：从依赖人类标注的偏好信号→可自动验证的规则化奖励（rule-based rewards）→Agent自监控奖励黑客。这使得RL训练规模可以脱离人类标注瓶颈，实现真正的scaling。

### 6.5 上下文长度演进：2K → 1M（500x跨越）

| 代次 | 原生上下文 | 推理扩展 | 关键技术 |
|------|----------|---------|---------|
| Qwen | 2K/8K | 32K | NTK-aware + LogN-Scaling + Window Attn |
| Qwen1.5 | 32K | 32K | 统一支持 |
| Qwen2 | 32K | 128K | RoPE base=1M + DCA + YARN |
| Qwen2.5 | 128K | 1M (Turbo/Plus) | 专门长上下文训练 |
| Qwen3 | 128K | 4x扩展 | YARN + DCA |
| Qwen3.5 | **262K原生** | — | Hybrid Attention效率支撑 |
| Qwen3-Coder 480B | — | **1M (YaRN)** | 代码超长项目理解 |
| Qwen3.7-Max | **1M原生** | — | Hybrid Attention + Agent级长上下文 |

上下文长度的指数增长是Qwen演进的**第二条最显著主线**。Hybrid Attention的引入是百万级原生上下文的技术前提。

### 6.6 数据规模即能力上限

| 代次 | 预训练数据 | 涨幅 | 主要能力提升 |
|------|----------|------|------------|
| Qwen/1.5 | 3T | — | 中英基础能力 |
| Qwen2 | 7T | +133% | 多语言（30种）、代码数学增强 |
| Qwen2.5 | 18T | +157% | 常识/专家知识/推理 |
| Qwen3 | 36T | +100% | 119种语言、STEM深度推理 |
| Qwen3.5 | 36T+ | — | **201种语言**（语种维度+70%） |

每代约翻倍的数据规模扩展是Qwen系列持续进步的第一驱动力。Qwen3.5开始转向**语言覆盖广度**的扩展。

### 6.7 设计哲学与技术趋势总结

#### 哲学一：架构简洁性优先（贯穿全系列）

- 选择GQA而非更复杂的MQA或Flash-Attention变体
- VL融合从Cross-Attention简化为MLP投影
- MoE架构两次重塑都朝向更简单的均衡策略
- Thinking模式通过简单的chat template控制（`/think`），无需结构修改

#### 哲学二：多模态以LLM为中心

所有多模态扩展均采用"编码器→适配器→LLM"统一范式：
- 视觉：ViT → MLP Merger → LLM
- 音频：Whisper → 直接注入 → LLM
- 全模态：多编码器 → 统一表示 → Thinker (LLM) → Talker

LLM基座升级自动带动所有模态能力提升。

#### 哲学三：从对话到Agent（2025–2026战略转向）

- **架构层面**：Hybrid Attention（Gated DeltaNet）替代Full Attention，超长上下文高效推理
- **能力层面**：从对话式AI转向长程自主Agent（Qwen3.7-Max的35小时持续运行）
- **多模态层面**：视觉+语言+音频+工具统一为一体化Agent基座（Qwen3.7-Plus的"看、想、写、做、验"闭环）
- **生态层面**：开源（Apache 2.0）+ 闭源API双轨并行，覆盖从本地部署（Coder-Next 3B激活）到云端旗舰（Qwen3.7-Max 1M上下文）

---

## 7. 值得关注的关键创新点（按重要性排列）

以下创新按**对Qwen系列乃至业界的深远影响**排序。

### 7.1 Thinker-Talker架构（Qwen2.5-Omni, 2025.03）— ★★★★★

**技术意义**：业界首个端到端全模态理解+流式语音生成架构。Thinker处理多模态输入并产生隐藏表示，Talker直接基于该高维语义进行双轨自回归生成，避免了级联TTS方案的语义损失与延迟。

**影响**：成为后续所有Omni模型（Qwen3-Omni、Qwen3.5-Omni）的架构母版；启发了实时同声传译Qwen3.5-LiveTranslate的设计；TMRoPE位置编码奠定了多模态时间对齐的基本范式。

### 7.2 Hybrid Attention（Qwen3-Next, 2025.09）— ★★★★★

**技术意义**：用Gated DeltaNet（线性注意力）+ Gated Attention替代标准Full Attention。在长序列输入下推理开销大幅低于Full Attention，同时保留全注意力的表达能力。

**影响**：使Qwen3.5的262K原生上下文与Qwen3.7-Max的1M原生上下文成为可能；Qwen3-Coder-Next将其引入代码方向，实现80B模型在3B激活下达到Sonnet 4.5级编码性能；Qwen3.5吞吐相比Qwen3-Max最高提升19倍。

### 7.3 Thinking/Non-Thinking双模式（Qwen3, 2025.04）— ★★★★★

**技术意义**：业界首个将推理模式和快速响应模式融合在**单一模型**中的开源大模型。通过chat template中`/think`和`/no_think`控制，支持精细的Thinking Budget预算。

**影响**：Qwen3-VL-8B-Thinking将推理范式扩展到VL；Qwen3.7-Max的35小时长程Agent执行从根本上依赖于Thinking能力。

### 7.4 M-RoPE / TMRoPE / Interleaved-MRoPE（Qwen2-VL起, 2024.06）— ★★★★★

**技术意义**：将RoPE频率维度均分为三个独立子空间（Temporal/Height/Width），实现文本/图像/视频位置编码的统一。TMRoPE进一步实现音频-视频时间对齐，Interleaved-MRoPE支持原生交错长序列。

**影响**：成为多模态位置编码的标杆方案，被广泛参考；使Qwen3-VL的256K原生交错图文长序列成为可能。

### 7.5 GRPO + Rule-based Rewards（Qwen3, 2025.04）— ★★★★

**技术意义**：基于规则验证的Group Relative Policy Optimization，无需训练独立的奖励模型，使RL规模脱离人类标注瓶颈。3,995组query-verifier对将Qwen3-235B-A22B在AIME'24从70.1提升至85.1（170步RL）。

**影响**：成为后续Qwen3.5/3.6/3.7的标准对齐方法；R1-Zero风格RLHF继承并放大该路线；Qwen3.7-Max的Agent自监控reward hacking是该路线的极致延伸。

### 7.6 Naive Dynamic Resolution（Qwen2-VL, 2024.06）— ★★★★

**技术意义**：业界首创"零归一化"动态分辨率方案。移除ViT绝对位置嵌入，改用2D-RoPE使ViT能处理任意尺寸输入。视觉token数与图像尺寸严格成正比，保留原始纵横比。

**影响**：被Qwen2.5-VL、Qwen3-VL继承并增强，成为现代VLM的标准做法。

### 7.7 Strong-to-Weak蒸馏（Qwen3, 2025.04）— ★★★★

**技术意义**：直接从大模型教师蒸馏output logits到小模型，仅需四阶段方法1/10的GPU时间，同时提升Pass@1（即时性能）和Pass@64（探索能力）。

**影响**：使Qwen3小模型矩阵（0.6B/1.7B/4B/8B/14B/32B）能够快速跟随基座升级；为后续Qwen3.5/3.6小尺寸变体提供方法论基础。

### 7.8 Window Attention混合ViT（Qwen2.5-VL, 2024.10）— ★★★★

**技术意义**：ViT大部分层使用窗口≤8×8的Window Attention，仅4层使用Full Attention，将复杂度从O(n²)降至接近O(n)。

**影响**：使高分辨率原生图像处理在工程上可行；ViT与LLM的归一化（RMSNorm）和激活函数（SwiGLU）统一，简化了多模态模型的工程栈。

### 7.9 Naive Dynamic Resolution → 绝对时间编码（Qwen2.5-VL, 2024.10）— ★★★★

**技术意义**：M-RoPE的temporal维度ID从帧索引升级为真实秒数，配合动态FPS训练，使模型真正理解视频中的时间流逝；Bbox采用绝对像素坐标。

**影响**：支持小时级长视频理解和秒级事件定位（`mm:ss.ff`格式），为视频Agent场景奠定基础。

### 7.10 长程Agent自监控（Qwen3.7-Max, 2026.05）— ★★★★

**技术意义**：在86小时RL训练中模型自主标记1,618个奖励黑客（reward hacking）案例；连续35小时自主运行，跨Claude Code/OpenClaw/Qwen Code等不同框架泛化。

**影响**：定义了"Agent Frontier"的新基准；将LLM能力衡量从单步对话延伸至长程任务执行。

### 7.11 BBPE 151K大词表（Qwen初代, 2023.08）— ★★★

**技术意义**：151,936词表（相比LLaMA 32K词表），中文编码效率提升3-4倍。

**影响**：为整个系列的多语言能力奠基，词表设计延续至Qwen3.5（虽微调至151,669，但量级保持）。

### 7.12 QK-Norm替代QKV Bias（Qwen3, 2025.04）— ★★★

**技术意义**：移除延续四代的QKV Bias设计，以归一化方案提升大规模训练稳定性。

**影响**：成为后续所有Qwen3.x系列的标准配置。

### 7.13 自然语言提示替代标签系统（Qwen2-Audio, 2024.07）— ★★★

**技术意义**：用自然语言提示替代Qwen-Audio的层次化标签系统（Transcription Tag/Task Tag等六层标签），相比标签系统提供更好的泛化与指令跟随能力。

**影响**：成为Qwen2.5-Omni端到端联合训练的方法论前驱。

### 7.14 Native MTP（Qwen3-Next, 2025.09）— ★★★

**技术意义**：在原生预训练中内置Multi-Token Prediction目标，提升训练效率和推理质量。

**影响**：被Qwen3.5/3.6/3.7全面继承。

---

## 8. 参考文献

### 8.1 基座语言模型

| # | 论文 | arXiv / 来源 | 时间 |
|---|------|-------------|------|
| 1 | Qwen Technical Report | arXiv:2309.16609 | 2023.09 |
| 2 | Qwen1.5 Blog | qwenlm.github.io/blog/qwen1.5/ | 2024.02 |
| 3 | Qwen2 Technical Report | arXiv:2407.10671 | 2024.07 |
| 4 | Qwen2.5 Technical Report | arXiv:2412.15115 | 2024.12 |
| 5 | Qwen3 Technical Report | arXiv:2505.09388 | 2025.05 |
| 6 | Qwen3-Next Blog | qwen.ai (2025.09) | 2025.09 |
| 7 | Qwen3.5 Blog | qwen.ai/blog?id=qwen3.5 | 2026.02 |
| 8 | Qwen3.7 The Agent Frontier | qwen.ai/blog?id=qwen3.7 | 2026.05 |

### 8.2 视觉语言模型

| # | 论文 | arXiv | 时间 |
|---|------|-------|------|
| 9 | Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond | arXiv:2308.12966 | 2023.08 |
| 10 | Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution | arXiv:2409.12191 | 2024.09 |
| 11 | Qwen2.5-VL Technical Report | arXiv:2502.13923 | 2025.01 |
| 12 | Qwen3-VL Technical Report | arXiv:2511.21631 | 2025.11 |
| 13 | Qwen3.7-Plus Blog | qwen.ai/blog?id=qwen3.7-plus | 2026.06 |

### 8.3 音频与全模态模型

| # | 论文 | arXiv | 时间 |
|---|------|-------|------|
| 14 | Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models | arXiv:2311.07919 | 2023.11 |
| 15 | Qwen2-Audio Technical Report | arXiv:2407.10759 | 2024.07 |
| 16 | Qwen2.5-Omni Technical Report | arXiv:2503.20215 | 2025.03 |
| 17 | Qwen3-Omni Blog | qwen.ai (2025.05) | 2025.05 |
| 18 | Qwen3.5-Omni Technical Report | arXiv:2604.15804 | 2026.04 |
| 19 | Qwen3.5-LiveTranslate-Flash Blog | qwen.ai (2026.05) | 2026.05 |

### 8.4 专业方向模型

| # | 论文 | arXiv / 来源 | 时间 |
|---|------|-------------|------|
| 20 | Qwen2.5-Coder Technical Report | arXiv:2409.12186 | 2024.09 |
| 21 | Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement | arXiv:2409.12122 | 2024.09 |
| 22 | QwQ-32B: Embracing the Power of Reinforcement Learning | Qwen Blog | 2025.03 |
| 23 | Qwen3-Coder 480B-A35B Blog | qwen.ai | 2025.07 |
| 24 | Qwen3-Coder-Next Blog | qwen.ai/blog?id=qwen3-coder-next | 2026.05 |

### 8.5 关键技术依赖

| # | 论文 | 来源 | 时间 |
|---|------|------|------|
| 25 | Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | ICML 2023 | 2023 |

### 8.6 官方资源

- Qwen官方网站：https://qwen.ai
- GitHub组织：https://github.com/QwenLM
- Hugging Face模型库：https://huggingface.co/Qwen
- Qwen1.5发布博客：https://qwenlm.github.io/blog/qwen1.5/
- Qwen3发布博客：https://qwenlm.github.io/blog/qwen3/
- Qwen3.5发布博客：https://qwen.ai/blog?id=qwen3.5
- Qwen3.7 Agent Frontier：https://qwen.ai/blog?id=qwen3.7
- Qwen3.7-Plus：https://qwen.ai/blog?id=qwen3.7-plus
- Qwen3-Coder-Next：https://qwen.ai/blog?id=qwen3-coder-next
- Qwen2-VL博客：https://qwenlm.github.io/blog/qwen2-vl/
- Qwen2.5-VL博客：https://qwenlm.github.io/blog/qwen2.5-vl/
- Qwen3-VL GitHub：https://github.com/QwenLM/Qwen3-VL

### 8.7 团队动态（2026年）

- 2026年1月：Qwen模型在HuggingFace累计下载量突破7亿
- 2026年3月：技术负责人林俊杨离职（曾主导Qwen3-Max开发）
- 2026年5月：团队由AI语音交互负责人Steven Hoi领导，定位"Foundation Models for the Agent Era"

---

*报告完*
