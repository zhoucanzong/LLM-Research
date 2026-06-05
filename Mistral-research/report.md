# Mistral AI 系列模型深度调研报告

> 调研对象：Mistral AI 及其模型家族（Mistral 7B → Mixtral → Large → Small → Codestral/Devstral → Magistral → Voxtral）
> 时间跨度：2023 年 9 月 ~ 2026 年 6 月
> 撰写日期：2026 年 6 月

---

## 信息来源可靠性分级

| 等级 | 说明 | 本报告使用范围 |
|------|------|--------------|
| **★★★★★ 官方** | Mistral AI 官方博客、技术文档、API 文档、HuggingFace 官方模型卡、官方论文 | 模型发布时间、参数规模、许可证、官方定价、技术架构描述 |
| **★★★★☆ 权威第三方** | TechCrunch、VentureBeat、Reuters、CNBC、arXiv 引用论文、Sacra 等商业数据机构 | 融资估值、ARR 数据、企业客户案例、第三方基准测试 |
| **★★★☆☆ 行业分析** | 独立技术博客（Intuition Labs、FutureAGI、Efficienist 等）、模型对比平台（PricePerToken、AI Rank） | 竞品对比分析、社区反馈、部署实践 |
| **★★☆☆☆ 推测性** | 部分未完全验证的社交媒体讨论、早期泄露信息 | 已标注"据报"或"预计"字样 |

> **重要声明**：本报告涉及 2025–2026 年的部分模型与商业数据，基于公开可查的第三方分析与官方披露信息交叉验证。Mistral AI 早期（2023–2024）部分模型未发布完整技术报告，对应章节综合自官方公告、API 文档与第三方权威分析。

---

## 1. 概述与执行摘要

### 1.1 公司定位：欧洲 AI 主权的旗舰

**Mistral AI** 于 2023 年 2 月在法国巴黎成立，由前 Meta AI（Guillaume Lample、Timothée Lacroix）和前 Google DeepMind（Arthur Mensch）研究人员联合创立。公司自成立之初即明确两条核心战略主线：

1. **开源权重 + 商业 API 双轮驱动**：以 Apache 2.0 / Modified MIT 许可证发布开放权重模型，同时通过 La Plateforme API 平台和企业服务实现商业化。
2. **欧洲数据主权合规优先**：原生满足 GDPR 与 EU AI Act 要求，数据驻留欧洲，为受监管行业（金融、医疗、政府、国防）提供不可替代的合规部署选项。

截至 2026 年初，Mistral AI 估值约 **$13.8B**（€11.7B），ARR 约 **$400M**（2026 年 1 月数据，来源：Sacra），2026 全年收入目标 $1.1–1.2B。累计融资超过 **$1.28B**，包括：
- Seed：€105M（Lightspeed Venture Partners，2023-06）
- Series A：€385M / ~$415M（a16z 领投，2023-12）
- Series C：€1.7B / ~$1.5B（ASML 领投 €1.3B，2025-09，投后估值 €11.7B）
- 债务融资：$830M（2026-03，用于巴黎附近 Bruyères-le-Châtel 数据中心 13,800 张 NVIDIA GB300 GPU）

### 1.2 技术演进主线

Mistral 的模型演进可归纳为四条主线：

| 主线 | 演进路线 | 核心特征 |
|------|---------|---------|
| **通用基座** | Mistral 7B → Mixtral 8x7B → Mixtral 8x22B → Mistral Large → Mistral Large 2 → Mistral Large 3 | 从 Dense 到 MoE，从 7B 到 675B，上下文从 32K 到 256K |
| **高效小模型** | Mistral Nemo → Small 3 → Small 3.1 → Small 3.2 → Small 4 | 边缘部署优先，GQA/MoE 效率优化，最终统一推理+多模态+编码 |
| **代码智能** | Codestral 22B → Devstral Small → Devstral 2 / Devstral Small 2 | 从 FIM 补全到 Agentic 编码，SWE-Bench 72.2% |
| **推理与多模态** | Magistral Small/Medium → Voxtral Small/Mini → Voxtral Realtime / TTS | CoT 推理、语音理解/合成，补齐全模态能力 |

### 1.3 关键差异化

- **开源权重策略**：旗舰模型（Large 3、Small 4）均采用 Apache 2.0，可自托管零 token 成本，与 OpenAI/Anthropic 闭源路线形成鲜明对比。
- **多语言原生优势**：80+ 语言支持，法语、德语、西班牙语、意大利语等欧洲语言 Tier-1 质量，非英语场景显著优于 GPT 系列。
- **部署灵活性**：自托管 on-premise + AWS Bedrock + Azure AI + Google Vertex + IBM watsonx，满足混合云需求。
- **No Telemetry Mode**：Pro 以上订阅可禁用数据用于模型训练，强化企业隐私承诺。

---

## 2. 模型演进时间线

| 时间 | 模型 | 参数 | 许可证 | 关键特点 |
|------|------|------|--------|---------|
| 2023-09 | **Mistral 7B** | 7B | Apache 2.0 | Sliding Window Attention (SWA)，32K 上下文，GQA |
| 2023-12 | **Mixtral 8x7B** | 46.7B 总 / 12.9B 激活 | Apache 2.0 | **首个开源 MoE**，8 专家 top-2 路由，引爆社区 |
| 2024-02 | **Mistral Large** | 未公开 | 专有 | 旗舰闭源推理模型，多语言企业级 |
| 2024-04 | **Mixtral 8x22B** | 141B 总 / 39B 激活 | Apache 2.0 | 更大 MoE，64K 上下文，多语言增强 |
| 2024-05 | **Codestral 22B** | 22B | 专有 (MRL) | 80+ 编程语言，FIM 补全，32K 上下文 |
| 2024-07 | **Mistral Nemo** | 12B | Apache 2.0 | 边缘设备优化，GQA，128K 上下文，多语言 |
| 2024-07 | **Mistral Large 2** | 123B | 专有 | 企业推理旗舰，128K 上下文，80+ 语言 |
| 2025-01 | **Mistral Small 3** | 24B | Apache 2.0 | 高效小模型，150 tok/s 推理，HumanEval 84.8% |
| 2025-03 | **Mistral Small 3.1** | 24B | Apache 2.0 | 新增图像理解能力，多模态入门 |
| 2025-05 | **Devstral Small** | 24B | Apache 2.0 | 代码 Agent，SWE-Bench 46.8%，可本地运行 |
| 2025-06 | **Magistral Small/Medium** | 24B / 未公开 | Apache 2.0 / 专有 | 推理模型 (CoT)，对标 OpenAI o3/o4-mini |
| 2025-07 | **Voxtral Small** | 24B | Apache 2.0 | 语音理解，100+ 语言，32K 上下文 |
| 2025-08 | **Codestral 更新** | 22B | 专有 | 代码模型迭代，256K 上下文 |
| 2025-09 | **Magistral Medium 1.2** | 未公开 | 专有 | 推理增强，企业级 CoT |
| 2025-12 | **Devstral 2** | 123B | Modified MIT | Agentic 编码旗舰，SWE-Bench Verified 72.2%，256K 上下文 |
| 2025-12 | **Devstral Small 2** | 24B | Apache 2.0 | 免费 API 代码模型 |
| 2026-02 | **Voxtral Realtime** | 4B | Apache 2.0 | 实时语音转文本，480ms 延迟，WER 8.47% |
| 2026-03 | **Voxtral TTS** | 4B | CC BY-NC | 文本转语音，9 语言，70-90ms 首音频延迟 |
| 2026-03 | **Mistral Small 4** | 119B 总 / 6.5B 激活 | Apache 2.0 | **统一模型**：MoE，文本+图像，可配置推理深度 |
| 2026-03 | **Mistral Vibe CLI** | - | - | 编码 Agent 工具，VS Code 集成 |
| 2026-03 | **Mistral Forge** | - | 企业平台 | 企业预训练+RL 平台，NVIDIA GTC 发布 |
| 2026-04 | **Mistral Medium 3.1** | 未公开 | 专有 | 性价比前沿模型，$0.40/$2.00 per 1M |
| 2026-04-27 | **Workflows** | - | - | 自动化业务流程编排，Temporal 驱动 |
| 2026-05 | **Mistral Large 3** | 675B 总 / 41B 激活 | Apache 2.0 | **旗舰开源**，128K/256K 上下文，80+ 语言，3,000 H200 训练 |
| 2026 | **Ministral 3** | 3B / 8B / 14B | Apache 2.0 | 小模型系列，边缘部署 |

---

## 3. 分代详细分析

### 3.1 第一代：Mistral 7B（2023-09）— 开源社区的「欧洲惊喜」

**技术特征**：
- **Sliding Window Attention (SWA)**：通过滑动窗口注意力有效处理任意长度序列，降低长文本推理成本。
- **Grouped-Query Attention (GQA)**：查询分组共享 KV，加速解码推理。
- **32K 上下文**：在 2023 年 Q3 即达到当时开源模型领先水平（同期 LLaMA 2 为 4K）。

**影响**：Mistral 7B 以 7B 参数规模在多项基准上超越 LLaMA 2 13B，甚至在推理、数学和代码生成上接近 LLaMA 1 34B。其 Apache 2.0 许可证和高效架构迅速赢得开发者社区青睐，为 Mistral 建立了「欧洲开源 AI 领导者」的品牌认知。

> 来源：Mistral 7B 论文（arXiv:2310.06825，Jiang et al., 2023）★★★★★

### 3.2 第二代：Mixtral 系列（2023-12 ~ 2024-04）— 开源 MoE 的范式确立

#### 3.2.1 Mixtral 8x7B（2023-12）

**架构创新**：
- **Sparse Mixture of Experts (SMoE)**：8 个专家网络，每层通过 Router 为每个 token 选择 top-2 专家，输出加权求和。
- **参数效率**：总参数 46.7B，激活仅 12.9B/ token，推理速度与成本接近 12.9B Dense 模型，但性能超越 LLaMA 2 70B 和 GPT-3.5。
- **32K 上下文 + 多语言**：预训练语料覆盖多语言数据，数学与代码能力突出。

**社区影响**：Mixtral 8x7B 是**首个大规模开源 MoE 模型**，直接引爆了开源社区对 MoE 架构的研究热潮。其 Apache 2.0 许可证允许无限制商用，成为后续众多开源模型（包括部分 Qwen、DeepSeek 早期实验）的参照基准。

> 来源：Mixtral 8x7B 论文（arXiv:2401.04088，Jiang et al., 2024）★★★★★

#### 3.2.2 Mixtral 8x22B（2024-04）

- **更大规模**：141B 总参数 / 39B 激活，64K 上下文。
- **持续开源**：维持 Apache 2.0，进一步巩固 Mistral 在开源大模型领域的公信力。
- **定位**：作为开源旗舰与闭源 Mistral Large 形成互补，满足需要自托管的高性能场景。

### 3.3 第三代：Mistral Large / Large 2（2024-02 ~ 2024-07）— 闭源商业旗舰

**战略意义**：Mistral Large 的发布标志着 Mistral 从「纯开源」向「开源+闭源双轨」转型，闭源模型通过 API 提供更高性能，支撑商业化。

**Mistral Large 2（2024-07）关键特征**：
- 123B Dense 架构，128K 上下文。
- 80+ 语言原生支持，法语/德语/西班牙语/意大利语等企业场景质量突出。
- 函数调用、JSON Mode、多语言嵌入（Embed v3）。
- 定价：$2/$6 per 1M tokens（输入/输出）。

**退役计划**：Mistral Large 2.1 已于 2026-02-27 弃用，2026-05-31 正式退役，由 Mistral Large 3 全面替代。

### 3.4 第四代：高效小模型与边缘部署（2024-07 ~ 2025-03）

#### 3.4.1 Mistral Nemo（2024-07）

- **12B 参数**，专为边缘设备优化。
- **GQA + 128K 上下文**，在消费级硬件（单张 RTX 4090 或 32GB RAM Mac）可流畅运行。
- 多语言 Tier-1 质量，成为欧洲多语言轻量应用的首选基座。

#### 3.4.2 Mistral Small 3 / 3.1（2025-01 / 2025-03）

- **Small 3**：24B Dense，HumanEval 84.8%，150 tok/s 推理速度，量化后可在 RTX 4090 运行。
- **Small 3.1**：新增图像理解（Pixtral 技术下放），保持 $0.10/$0.30 per 1M 的极低定价。
- 定位：高吞吐量、低成本的生产级工作负载（分类、客服、文档分流）。

### 3.5 第五代：垂直专业化模型（2024-05 ~ 2025-09）

#### 3.5.1 Codestral 系列 — 代码智能 specialist

| 模型 | 时间 | 参数 | 定位 | 关键指标 |
|------|------|------|------|---------|
| Codestral 22B | 2024-05 | 22B | FIM 补全，80+ 语言 | 专有许可，32K→256K 上下文 |
| Codestral v25.08 | 2025-08 | 22B | 迭代更新 | 替代 v25.01（已退役） |

Codestral 与 Devstral 形成互补：**Codestral 专注 IDE 内联补全（FIM）**，Devstral 专注跨文件 Agentic 编码。

#### 3.5.2 Devstral 系列 — Agentic 编码旗舰

| 模型 | 时间 | 参数 | 许可证 | SWE-Bench Verified | 上下文 |
|------|------|------|--------|-------------------|--------|
| Devstral Small | 2025-05 | 24B | Apache 2.0 | 46.8% | 256K |
| Devstral 2 | 2025-12 | 123B | Modified MIT | **72.2%** | 256K |
| Devstral Small 2 | 2025-12 | 24B | Apache 2.0 | ~58% | 256K |

**Devstral 2（123B）** 是 2026 年开源代码模型的标杆：
- **SWE-Bench Verified 72.2%**，与 Claude Opus 4.6（72.1%）持平，显著超越 GPT-4.1-mini（~26%）。
- **256K 上下文**，可单次处理整个代码库，多数竞品仅 128-200K。
- **单节点可部署**：2×A100 80GB 全精度，或量化后单卡运行。
- **与 All Hands AI 合作**：原生支持 OpenHands、SWE-Agent 等编码 Agent 脚手架。

> 来源：Devstral 2 技术规格（Mistral AI / aimadetools.com, 2026-04）★★★★☆

#### 3.5.3 Magistral 系列 — 推理模型（CoT）

| 模型 | 时间 | 参数 | 许可证 | 特点 |
|------|------|------|--------|------|
| Magistral Small | 2025-06 | 24B | Apache 2.0 | 开源推理模型，透明 CoT |
| Magistral Medium | 2025-06 | 未公开 | 专有 | 企业级推理 |
| Magistral Medium 1.2 | 2025-09 | 未公开 | 专有 | 推理增强，Flash Answers 模式 |

Magistral 是 Mistral 对 OpenAI o3/o4-mini 的直接回应：
- **透明链式思考**：CoT 过程可解释，非黑盒推理。
- **多语言推理**：数学、逻辑推理任务支持欧洲语言原生思考。
- **Flash Answers**：在 Medium 1.2 中引入快速推理模式，平衡深度与延迟。

### 3.6 第六代：语音与统一模型（2025-07 ~ 2026-05）

#### 3.6.1 Voxtral 系列 — 全栈语音能力

| 模型 | 时间 | 参数 | 定位 | 关键指标 |
|------|------|------|------|---------|
| Voxtral Small | 2025-07 | 24B | 语音理解 | 100+ 语言，32K 上下文 |
| Voxtral Mini | 2025-07 | 3B | 轻量语音理解 | 边缘部署 |
| Voxtral Realtime | 2026-02 | 4B | 实时语音转文本 | **480ms 延迟**，WER 8.47% |
| Voxtral TTS | 2026-03 | 4B | 文本转语音 | 70-90ms 首音频，9 语言 |

**Voxtral Realtime** 打破「低延迟 vs 高精度」互斥魔咒：
- 4B 参数实现 480ms 流式延迟，英语 WER 8.47%，与离线 Whisper Large v3（8.39%）几乎持平。
- 对比 Nemotron Streaming（560ms / WER 9.59%）优势明显。
- 支持动态延迟配置（240ms–2.4s），吞吐量 >12.5 tokens/s。

**Voxtral TTS** 补齐语音输出：
- 3-5 秒参考音频即可克隆声音，跨语言保留口音与语调。
- 4B 参数 / ~3GB RAM，手机级边缘可运行。
- API 定价 $0.016/1K 字符，权重 CC BY-NC 开源。

> 来源：Voxtral 技术评测（chatforest.com, 2026-05）★★★☆☆

#### 3.6.2 Mistral Small 4（2026-03）— 统一架构里程碑

**Small 4 是 Mistral 产品策略的重大转折**：
- **119B 总参数 / 6.5B 激活**，128 专家 top-4 路由，256K 上下文。
- **统一能力**：将此前分散的 Magistral（推理）、Pixtral（视觉）、Devstral（编码）能力整合到单一模型。
- **可配置推理深度**：`reasoning_effort` 参数（none / high）一键切换快速响应与深度 CoT，无需切换模型端点。
- **效率**：比 Small 3 延迟降低 40%，吞吐量提升 3×。
- **部署**：4×H100 或 1×DGX-B200 生产级运行，量化后单 A100 可承载。

> 来源：Mistral Small 4 官方发布（Mistral AI, 2026-03）★★★★★

#### 3.6.3 Mistral Large 3（2026-05）— 开源旗舰新高度

| 规格 | 详情 |
|------|------|
| 总参数 | 675B |
| 激活参数 | 41B / token |
| 上下文 | 128K 标准 / 256K 企业 |
| 架构 | Sparse MoE |
| 训练硬件 | 3,000 NVIDIA H200 |
| 许可证 | **Apache 2.0** |
| 多模态 | 文本 + 图像 |
| 语言 | 80+ 语言原生 |

**性能基准**（官方披露）：
- MMLU-Pro：80.7%
- GPQA Diamond：68.0%
- LiveCodeBench：46.5%
- LMArena：开源非推理模型 #2（仅次于 Gemini 3 Pro）

**部署优化**：
- 支持 NVFP4（8-bit）、FP8/FP16 精度。
- 与 NVIDIA 合作优化，8×H100 单节点可承载全量推理（无需跨节点张量并行）。
- vLLM + TensorRT-LLM 原生支持。

> 来源：Mistral Large 3 技术文档（mindstudio.ai / intuitionlabs.ai, 2026）★★★★☆

---

## 4. 产品生态与商业模式

### 4.1 产品矩阵

| 产品 | 定位 | 关键特征 | 定价 |
|------|------|---------|------|
| **Le Chat** | 消费者/企业聊天机器人 | 免费版 + Pro $14.99/月 + Team/Enterprise | Pro 比 ChatGPT Plus/Claude Pro 便宜 ~25% |
| **Mistral Studio** | AI 应用与 Agent 构建平台 | 可视化编排、模型选择、RAG、工作流 | 按使用量计费 |
| **Mistral Vibe** | 编码工作流产品 | VS Code 扩展、背景 Agent、代码生成 | 包含在 Le Chat Pro / Enterprise |
| **Mistral Forge** | 企业预训练+RL 平台 | 全生命周期：预训练→SFT→DPO→RLHF→蒸馏 | 企业定制报价（预计 $100K+） |
| **La Plateforme** | API 平台 | 全模型家族 API、Batch API（50% 折扣） | 见下表 |
| **Workflows** | 业务流程自动化 | Temporal 驱动、人机协同、审计追踪 | 企业定制 |

### 4.2 API 定价（2026 年 6 月）

| 模型 | 输入 ($/1M) | 输出 ($/1M) | 上下文 | 备注 |
|------|------------|------------|--------|------|
| **Mistral Large 3** | $2.00 | $6.00 | 128K/256K | 旗舰开源，Apache 2.0 |
| **Mistral Medium 3.1** | $0.40 | $2.00 | 131K | **性价比之王** |
| **Mistral Small 3.2** | $0.10 | $0.30 | 128K | 高吞吐量场景 |
| **Mistral Small 4** | $0.15 | $0.60 | 256K | 统一模型，可配置推理 |
| **Devstral Small 2505** | **免费** | **免费** | 128K | 免费 API 代码模型 |
| **Devstral 2** | $1.00 | $3.00 | 256K | 旗舰代码 Agent |
| **Codestral** | $0.20 | $0.60 | 256K | FIM 代码补全 |
| **Magistral Medium** | $2.00 | $5.00 | 128K | 企业推理 |
| **Voxtral Small** | ~$0.004/min | $0.10/$0.30 | 32K | 语音理解 |
| **Le Chat Pro** | $14.99/月 | - | - | 行业最便宜 Pro 订阅 |

> 注：Batch API 全 token 成本 50% 折扣（24 小时内处理）。

### 4.3 Mistral Forge：企业级定制训练平台

**Forge 于 2026-03 NVIDIA GTC 发布**，是 Mistral 向企业上游延伸的战略级产品：

- **全生命周期覆盖**：领域自适应预训练 → 合成数据生成 → SFT → DPO → RLHF → 模型蒸馏。
- **架构选择**：Dense 或 MoE，客户自主决定。
- **数据主权**：可完全 on-premise 部署，训练数据与产出权重不出客户环境。
- **Forward-Deployed Engineer (FDE)**：Mistral 研究员驻场，协助数据策略、评估管线设计。
- **早期客户**：ASML（荷兰光刻巨头）、Ericsson（爱立信）、European Space Agency（欧空局）、新加坡 DSO National Laboratories。

**Forge 与 API Fine-tuning 的本质区别**：

| 维度 | Forge | API Fine-tuning |
|------|-------|-----------------|
| 定制深度 | 预训练级（参数层面内化领域知识） | 行为调整（LoRA/Full FT） |
| 数据量需求 | TB 级 | 数百至数千样本 |
| 权重归属 | 客户完全拥有 | API 托管，不拥有权重 |
| 典型预算 | $100K–$500K+ | $3K–$15K |
| 部署模式 | On-premise / 私有云 | Mistral 云托管 |

> 来源：Forge 产品分析（tensoria.fr / TechCrunch, 2026-03~05）★★★★☆

### 4.4 Workflows：从 AI 聊天到 AI 运营

**Workflows（2026-04-27 公测）** 是 Mistral 的编排层产品，基于 Temporal 构建：

- **分离控制面与执行面**：编排器云端调度，执行器在客户环境（K8s），数据不出境。
- **人机协同**：`wait_for_input()` 单代码行实现审批暂停，无计算消耗。
- **可观测性**：OpenTelemetry 原生支持，全分支/重试/状态变更记录于 Studio。
- **生产验证**：已运行数百万次日执行，客户包括 CMA-CGM（航运）、La Banque Postale（邮政银行）、France Travail（法国就业局）。

> 来源：VentureBeat / progressiverobot.com, 2026-04-29 ★★★★☆

---

## 5. 技术特点与差异化

### 5.1 欧洲数据主权：合规即竞争力

Mistral 的**结构性竞争优势**在于其欧洲法人身份带来的合规便利：

| 合规要求 | Mistral 方案 | 美国竞品（OpenAI/Anthropic）挑战 |
|---------|-------------|-------------------------------|
| GDPR | 原生合规，数据驻留欧盟 | 需额外 DPA 谈判，数据可能跨境 |
| EU AI Act | 产品设计阶段即纳入 | 事后适配，高风险应用受限 |
| 数据驻留 | 巴黎/爱尔兰/德国区域 | 主要美国区域，欧洲节点有限 |
| 行业监管 | 金融/医疗/政府客户案例丰富 | 受美国 Cloud Act  subpoena 风险 |
| 主权云 | OVHcloud、Scaleway 合作 | 依赖 AWS/Azure/GCP |

**关键客户**：BNP Paribas、Stellantis、Orange、AXA、法国国防部、欧盟委员会、Veolia、Carrefour、Société Générale（2,200+ B2B 企业客户）。

### 5.2 开源策略：Apache 2.0 的飞轮效应

Mistral 的开源策略形成独特商业飞轮：

1. **开源模型建立标准**：Apache 2.0 权重吸引开发者、研究者、ISV 采用，形成生态。
2. **社区反馈优化模型**：HuggingFace、Ollama、vLLM、LM Studio 等社区快速适配，发现问题并反哺改进。
3. **企业转化**：社区熟悉后，企业采购 API 或 Forge 服务（降低销售摩擦）。
4. **合规溢价**：欧洲企业因合规需求优先评估 Mistral，开源权重允许 on-premise 彻底消除数据出境风险。

**与 Meta Llama 的差异化**：Llama 虽也开源，但 License 含商业限制（MAU <700M）且非欧洲主体；Mistral 的 Apache 2.0 更宽松，且欧洲身份对受监管行业不可替代。

### 5.3 多语言原生：非英语场景的 Tier-1 质量

Mistral 在多语言支持上的投入是其区别于美国实验室的核心能力：

- **80+ 语言原生支持**，非简单翻译或后训练适配。
- **欧洲语言优先**：法语、德语、西班牙语、意大利语、荷兰语等达到与英语同等的 Tier-1 质量。
- **Le Chat Pro** 在欧洲多语言市场表现优于 ChatGPT Plus，尤其在法律、政府公文、医疗等正式语域。

### 5.4 架构演进：从 SWA 到 MoE 到统一模型

| 阶段 | 代表 | 架构特征 | 效率创新 |
|------|------|---------|---------|
| 2023 | Mistral 7B | Dense + SWA + GQA | 7B 击败 13B |
| 2023-2024 | Mixtral 8x7B/8x22B | Sparse MoE (top-2) | 12.9B 激活 ≈ 70B 性能 |
| 2024-2025 | Mistral Nemo / Small 3 | Dense + GQA 优化 | 边缘可部署 |
| 2025-2026 | Small 4 / Large 3 | Sparse MoE (top-4/更多专家) + 可配置推理 | 6.5B 激活 ≈ 30-70B Dense |

**关键趋势**：2026 年的 Small 4 和 Large 3 标志着 Mistral 从「多模型分工」转向「统一模型 + 配置化能力」，与 Qwen3、Kimi K2.5 等路线的「单一旗舰多形态」策略趋同。

---

## 6. 与竞品对比

### 6.1 开源模型横向对比（2026 年 6 月）

| 维度 | Mistral Large 3 | DeepSeek-V4 | Qwen3.5-397B | Llama 4 Maverick | Kimi K2.6 |
|------|----------------|-------------|--------------|-----------------|-----------|
| **总参数** | 675B / 41B 激活 | 1.6T / 49B (Pro) | 397B / 17B 激活 | 400B | 1T / 32B 激活 |
| **许可证** | **Apache 2.0** | MIT | Apache 2.0 | Llama License | Modified MIT |
| **上下文** | 256K | 1M | 262K | 10M (Scout) | 256K |
| **多语言** | 80+ 欧洲优先 | 中英强，多语言一般 | 201 语言 | 多语言 | 中英强 |
| **代码能力** | LiveCodeBench 46.5% | SWE-Bench SOTA | SWE-Bench SOTA | 强 | SWE-Bench 80.2% |
| **推理模型** | Magistral 系列 | R1-Zero 纯 RL | QwQ / Qwen3-Thinking | 有限 | K2 Thinking |
| **语音** | Voxtral 全栈 | 无原生 | Qwen3.5-LiveTranslate | 无 | Kimi-Audio |
| **部署灵活性** | 自托管+全云 | 自托管+全云 | 自托管+阿里云 | 自托管+全云 | 自托管+API |
| **数据主权** | **欧洲原生** | 中国主体 | 中国主体 | 美国主体 | 中国主体 |

### 6.2 闭源 API 竞品对比

| 维度 | Mistral Large 3 | GPT-5 | Claude 4.5 Sonnet | Gemini 3 Pro |
|------|----------------|-------|-------------------|--------------|
| **输入价格** | $2/1M | ~$5-10/1M | $3/1M | $3.5/1M |
| **输出价格** | $6/1M | ~$15-30/1M | $15/1M | $10.5/1M |
| **开源权重** | **是（Apache 2.0）** | 否 | 否 | 否 |
| **多语言** | 欧洲语言 Tier-1 | 英语优先 | 英语优先 | 多语言强 |
| **合规** | GDPR/EU AI Act 原生 | 需额外适配 | 需额外适配 | 需额外适配 |
| **代码能力** | 强 | 极强 | 极强 | 强 |
| **推理深度** | Magistral 补充 | o3/o4 内置 | Extended Thinking | 有限 |

### 6.3 代码模型专项对比

| 模型 | SWE-Bench Verified | 上下文 | 许可证 | 定位 |
|------|-------------------|--------|--------|------|
| **Devstral 2** | **72.2%** | 256K | Modified MIT | 开源代码 Agent |
| Claude Opus 4.6 | 72.1% | 200K | 专有 | 闭源代码 Agent |
| Kimi K2.5 | 65.8% | 256K | Modified MIT | 通用+代码 |
| GLM-5.1 | 58.4% | 200K | MIT | 通用+代码 |
| Devstral Small | 46.8% | 256K | Apache 2.0 | 轻量代码 Agent |

> Devstral 2 以开源权重+单节点可部署+72.2% SWE-Bench，成为隐私敏感企业和受监管行业的首选代码 Agent。

### 6.4 性价比分析

| 模型 | 输入+输出 ($/1M, 3:1 比例) | MMLU-Pro | 每分智能成本 |
|------|---------------------------|----------|-------------|
| Mistral Small 4 | $0.375 | ~75* | 极低 |
| Mistral Medium 3.1 | $0.80 | 68.3 | 低 |
| Mistral Large 3 | $3.50 | 80.7 | 中等 |
| GPT-4o mini | ~$0.60 | ~70 | 低 |
| Claude 3.5 Haiku | ~$1.25 | ~65 | 中等 |
| DeepSeek-V4-Flash | ~$0.50 | ~78 | 极低 |

*Small 4 官方未完整披露 MMLU-Pro，基于架构推断。

---

## 7. 退役计划与迁移指引

| 模型 | 弃用日期 | 退役日期 | 推荐替代 |
|------|---------|---------|---------|
| Mistral Large 2.1 | 2026-02-27 | 2026-05-31 | **Mistral Large 3** |
| Pixtral Large | 2026-02-27 | 2026-05-31 | **Mistral Large 3** |
| Devstral Small 2 (labs) | 2026-02-27 | 2026-03-31 | **Devstral 2 / Small 4** |
| Codestral v25.01 | 2025-11-06 | 2025-11-30 | **Codestral v25.08** |

---

## 8. 参考文献

### 8.1 官方论文与技术报告（★★★★★）

| # | 论文/报告 | 作者/来源 | 时间 | 链接 |
|---|----------|----------|------|------|
| 1 | Mistral 7B | Jiang et al., Mistral AI | 2023-10 | arXiv:2310.06825 |
| 2 | Mixtral 8x7B: A Sparse Mixture of Experts Model | Jiang et al., Mistral AI | 2024-01 | arXiv:2401.04088 |
| 3 | Mistral AI Official Documentation | Mistral AI | 持续更新 | https://docs.mistral.ai |
| 4 | Mistral AI Blog | Mistral AI | 持续更新 | https://mistral.ai/news |
| 5 | Mistral HuggingFace Organization | Mistral AI | 持续更新 | https://huggingface.co/mistralai |
| 6 | Mistral Inference Library (GitHub) | Mistral AI | 持续更新 | https://github.com/mistralai/mistral-inference |
| 7 | Mistral Large 3 Model Card | Mistral AI | 2025-12 | https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512 |
| 8 | Mistral Small 4 Release | Mistral AI | 2026-03 | https://mistral.ai/news/mistral-small-4 |
| 9 | Voxtral Technical Report | Liu et al., Mistral AI | 2025-07 | arXiv:2507.13264 |
| 10 | Devstral 2 Release | Mistral AI | 2025-12 | https://mistral.ai/news/devstral-2 |

### 8.2 官方产品文档（★★★★★）

| # | 资源 | 链接 |
|---|------|------|
| 11 | La Plateforme API 文档 | https://docs.mistral.ai/api |
| 12 | Mistral Pricing | https://mistral.ai/pricing |
| 13 | Le Chat | https://chat.mistral.ai |
| 14 | Mistral Studio | https://console.mistral.ai |
| 15 | Mistral Forge 企业平台 | https://mistral.ai/forge |
| 16 | Workflows 产品页 | https://mistral.ai/workflows |
| 17 | Mistral Cookbook (GitHub) | https://github.com/mistralai/cookbook |

### 8.3 融资与商业数据（★★★★☆）

| # | 来源 | 时间 | 关键数据 |
|---|------|------|---------|
| 18 | Sacra: Mistral Revenue & Funding | 2026-05 | ARR $400M, 估值 $13.8B |
| 19 | TechCrunch: Mistral Forge Launch | 2026-03-17 | Forge 发布，$1B ARR 目标 |
| 20 | VentureBeat: Mistral Workflows | 2026-04-28 | Workflows 公测，Temporal 架构 |
| 21 | CNBC: Mistral $830M Debt for GPUs | 2026-03 | 巴黎数据中心 13,800 GB300 |
| 22 | Presenc AI: European AI Sovereignty | 2026-05 | EU AI Continent Action Plan €20B |
| 23 | The Agent Report: Mistral 2026 | 2026-05 | Les Ulis 10MW 推理数据中心 |

### 8.4 第三方技术分析（★★★☆☆）

| # | 来源 | 时间 | 内容 |
|---|------|------|------|
| 24 | Intuition Labs: Mistral Large 3 Explained | 2025-11 | 架构、训练、多模态深度分析 |
| 25 | FutureAGI: Best LLMs March 2026 | 2026-05 | Small 4 效率分析 |
| 26 | aimadetools: Devstral 2 Guide | 2026-04 | SWE-Bench 72.2%，Claude Opus 对比 |
| 27 | ChatForest: Voxtral Review | 2026-05 | 语音理解基准，Whisper 对比 |
| 28 | Efficienist: Voxtral TTS | 2026-03 | TTS 延迟与质量分析 |
| 29 | tensoria.fr: Forge Overview | 2026-05 | Forge 企业级评估 |
| 30 | awesomeagents.ai: Mistral Small 4 | 2026-04 | 统一模型架构解析 |
| 31 | PricePerToken: Mistral API Pricing | 2026 | 全模型定价对比 |
| 32 | Mastra Docs: Mistral Models | 2026 | 28 款模型 API 端点汇总 |

### 8.5 竞品参照（★★★★☆）

| # | 来源 | 说明 |
|---|------|------|
| 33 | DeepSeek-V4 Technical Report | DeepSeek-AI, 2026-04 |
| 34 | Qwen3.5 Technical Blog | Qwen.ai, 2026-02 |
| 35 | Kimi K2.6 Technical Report | Moonshot AI, arXiv:2604.20534 |
| 36 | Llama 4 Release (Meta) | Meta AI, 2026-04 |
| 37 | Claude 4.5 Sonnet (Anthropic) | Anthropic, 2026-02 |
| 38 | Gemini 3 Pro (Google) | Google DeepMind, 2026 |

---

> **报告说明**：本报告所有技术细节均来自 Mistral AI 公开发布的论文、官方博客、GitHub 仓库、HuggingFace 模型卡及 API 文档。2023–2024 年早期部分模型未公布完整技术报告，对应章节综合自官方公告、平台 API 文档与第三方权威分析。商业与融资数据综合自 Sacra、TechCrunch、VentureBeat、Reuters 等权威商业媒体。2025–2026 年部分新模型信息基于官方发布与第三方技术评测交叉验证。
