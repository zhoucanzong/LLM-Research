# Qwen多模态模型与生态全景

> 调研日期：2025年6月（2026年6月更新）  
> 覆盖范围：Qwen音频模型系列、Qwen-Omni全模态模型、专业方向模型（Coder/Math/QwQ）及完整生态

---

## Part 1：音频模型深度解析

### 1.1 Qwen-Audio（2023年11月）

#### 1.1.1 概述

Qwen-Audio是阿里巴巴Qwen团队于2023年11月发布的**通用音频理解大模型**，是首个通过大规模音频-语言预训练覆盖超过30种任务的统一音频理解模型。该模型突破了此前音频模型只能处理单一类型音频（如纯语音或纯环境音）的局限，实现了对人类语音、自然音、音乐和歌曲的统一理解。

**论文信息**：  
- 标题：*Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models*  
- arXiv: 2311.07919  
- 发布时间：2023年11月14日  
- 作者：Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, Jingren Zhou  
- 机构：Alibaba Group

#### 1.1.2 模型架构

Qwen-Audio采用经典的**编码器-解码器**架构，包含两个核心组件：

**音频编码器**：
- 初始化自 **Whisper-large-v2**（OpenAI的大规模语音识别模型）
- 将音频波形转换为高维特征表示
- 输入：原始音频波形
- 输出：音频特征序列，直接注入LLM的输入序列中

**大语言模型（LLM）**：
- 基于 **Qwen-7B** 初始化
- 32层Transformer解码器
- 隐藏维度：4096
- 总参数量：约7.7B

**音频-文本融合方式**：  
音频编码器的输出特征直接作为前缀token注入LLM的输入序列，模型以自回归方式生成文本输出。训练目标为最大化条件概率：

$$P_θ(x_t | x_{<t}, Encoder_φ(a))$$

其中 θ 为LLM参数，φ 为音频编码器参数。

#### 1.1.3 多任务预训练框架

Qwen-Audio的核心创新在于其**层次化标签多任务训练框架**，通过条件序列标签解决多任务联合训练的干扰问题：

| 标签层级 | 功能 | 示例 |
|---------|------|------|
| Transcription Tag | 区分转写/分析类任务 | `<\|startoftranscripts\|>`, `<\|startofanalysis\|>` |
| Audio Language Tag | 标识音频中的语言 | 8种语言标记 + `<\|unknown\|>` |
| Task Tag | 指定具体任务类别 | `<\|transcribe\|>`, `<\|translate\|>`, `<\|caption\|>`, `<\|analysis\|>`, `<\|question-answer\|>` |
| Text Language Tag | 指定输出文本语言 | 对应各语言标记 |
| Timestamps Tag | 是否预测时间戳 | `<\|timestamps\|>`, `<\|notimestamps\|>` |
| Output Instruction | 细化任务和输出格式 | 具体文本指令 |

**关键技术特色 — SRWT（Speech Recognition with Word-level Timestamps）**：
- 训练模型在识别语音的同时预测每个词的精确时间戳
- 时间戳与转写词交替预测（开始时间在词前，结束时间在词后）
- 实验证明SRWT显著提升了ASR和音频问答等下游任务性能

#### 1.1.4 预训练数据与任务覆盖

| 音频类型 | 任务 | 数据量（小时） |
|---------|------|--------------|
| 语音 | ASR（多语言语音识别） | 30,000 |
| 语音 | S2TT（语音翻译） | 3,700 |
| 语音 | SRWT（词级时间戳识别） | 21,000 |
| 语音 | 方言识别/ASR | 4,000 |
| 语音 | 语言识别（LID） | 11,700 |
| 语音 | 说话人性别/年龄识别 | 9,600 |
| 语音 | 情感识别/关键词检测/意图分类 | <1,000 |
| 自然音 | 音频描述（AAC） | 8,400 |
| 自然音 | 声音事件分类/检测 | 5,400+ |
| 自然音 | 声学场景分类 | <1,000 |
| 音乐/歌曲 | 音乐描述 | 25,000 |
| 音乐/歌曲 | 音乐流派识别 | 9,500 |
| 音乐/歌曲 | 歌手识别/乐器分类/音高分析 | <1,000 |

**总计覆盖超过30种音频任务**，训练数据超过14万小时。

#### 1.1.5 训练策略

| 配置项 | 多任务预训练 | 监督微调 |
|--------|------------|---------|
| 音频编码器初始化 | Whisper-large-v2 | 第一阶段模型 |
| LLM初始化 | Qwen-7B | Qwen-7B |
| 优化器 | AdamW | AdamW |
| 峰值学习率 | 5e-5 | 1e-5 |
| 训练步数 | 500K | 8K |
| Batch Size | 120 | 128 |
| 精度 | bfloat16 | bfloat16 |

**关键训练细节**：
- 预训练阶段：**冻结LLM，仅训练音频编码器**
- 微调阶段：**冻结音频编码器，仅训练LLM**

#### 1.1.6 主要性能

| 任务 | 数据集 | 指标 | Qwen-Audio | 前最佳 |
|------|--------|------|-----------|--------|
| ASR | LibriSpeech test-clean | WER↓ | 2.0% | 2.1% |
| ASR | Aishell1 test | WER↓ | 1.3% | 1.9% |
| S2TT | CoVoST2 en-zh | BLEU↑ | 41.5 | 33.1 |
| AAC | Clotho | SPIDEr↑ | 0.288 | 0.271 |
| ASC | CochlScene | ACC↑ | 79.5% | 66.9% |
| VSC | VocalSound | ACC↑ | 92.9% | 60.4% |
| MNA | NSynth Instrument | ACC↑ | 78.8% | 50.1% |

---

### 1.2 Qwen2-Audio（2024年7月）

#### 1.2.1 概述

Qwen2-Audio是Qwen-Audio的重大升级版本，发布于2024年7月。相比前代，Qwen2-Audio进行了多项关键改进：简化预训练流程、扩大数据规模、引入DPO对齐，并实现了双模式音频交互能力。

**论文信息**：  
- 标题：*Qwen2-Audio Technical Report*  
- arXiv: 2407.10759  
- 发布时间：2024年7月15日  
- 作者：Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, Jingren Zhou  
- 机构：Qwen Team, Alibaba Group  
- 总参数量：**8.2B**

#### 1.2.2 架构改进

**音频编码器升级**：
- 初始化自 **Whisper-large-v3**（相比前代的v2）
- 音频重采样至 **16kHz**
- 特征提取：**128通道梅尔频谱图**（window size=25ms, hop size=10ms）
- 新增**池化层**（stride=2），将音频表示长度减半
- 编码器输出每帧对应约 **40ms** 原始音频

**LLM基座**：仍为 Qwen-7B

#### 1.2.3 训练流程革新

Qwen2-Audio采用**三阶段训练**流程：

**Stage 1：预训练（自然语言提示）**
- 核心改变：**用自然语言提示替代层次化标签**
- 发现：自然语言提示相比标签系统能提供更好的泛化能力和指令跟随能力
- 大幅扩展数据量

**Stage 2：监督微调（SFT）**
- 精心筛选高质量SFT数据
- 严格质量控制流程
- 实现两种无缝集成的交互模式

**Stage 3：DPO（直接偏好优化）**
- 使用人类标注的好/坏回复对进行训练
- 优化模型在事实性和行为规范方面的表现
- 提升指令跟随能力

#### 1.2.4 双模式音频交互

| 模式 | 描述 | 典型场景 |
|------|------|---------|
| **语音聊天（Voice Chat）** | 用户通过语音自由对话，无需文本输入 | 在线实时交互 |
| **音频分析（Audio Analysis）** | 用户提供音频+文本指令进行分析 | 离线音频文件分析 |

**关键特性**：两种模式无需通过系统提示切换，模型能智能理解音频内容并识别其中的语音指令，在包含环境音、多人对话和语音命令的复杂音频中直接响应。

#### 1.2.5 性能对比

| 任务 | 数据集 | Qwen2-Audio | Qwen-Audio | Gemini-1.5-pro |
|------|--------|-------------|-----------|---------------|
| ASR | LibriSpeech test-clean | **1.6%** | 2.0% | - |
| ASR | LibriSpeech test-other | **3.6%** | 4.2% | - |
| ASR | Aishell2 Android | **2.9%** | 3.3% | - |
| S2TT | CoVoST2 en-zh | **45.2** | 41.5 | - |
| S2TT | CoVoST2 zh-en | **24.4** | 15.7 | - |
| VSC | VocalSound | **93.9%** | 92.9% | - |
| AIR-Bench | Speech | **7.18** | 6.47 | 6.97 |
| AIR-Bench | Sound | **6.99** | 6.95 | 5.49 |
| AIR-Bench | Music | **6.79** | 5.52 | 5.06 |
| AIR-Bench | Mixed | **6.77** | 6.08 | 5.27 |

Qwen2-Audio在AIR-Bench指令跟随评测中**全面超越Gemini-1.5-pro**，达到SOTA水平。

#### 1.2.6 多语言覆盖

Qwen2-Audio支持的语言包括但不限于：
- 英语、中文（普通话）、粤语
- 法语、德语、西班牙语、意大利语
- 以及Common Voice等数据集覆盖的更多语言

---

## Part 2：Qwen2.5-Omni全模态模型深度解析

### 2.1 概述

Qwen2.5-Omni是Qwen团队于2025年3月发布的**端到端全模态模型**，首次在单一模型中统一了文本、图像、音频和视频的感知能力，并能以流式方式同时生成文本和自然语音响应。

**论文信息**：  
- 标题：*Qwen2.5-Omni Technical Report*  
- arXiv: 2503.20215  
- 发布时间：2025年3月26日  
- 作者：Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, Junyang Lin  
- 参数规模：**7B**

### 2.2 Thinker-Talker 架构

Qwen2.5-Omni的核心架构创新是 **Thinker-Talker** 双模块设计，灵感来源于人脑的"思考"与"表达"分离：

#### 2.2.1 Thinker（思考者）

- 功能：如同"大脑"，负责接收和理解多模态输入，生成文本响应
- 本质：一个大型语言模型
- 输入：文本、图像、音频、视频的统一表示
- 输出：文本token + 高维隐藏表示（传递给Talker）

#### 2.2.2 Talker（表达者）

- 功能：负责语音生成
- 架构：**双轨自回归Transformer解码器**（设计灵感源自Mini-Omni）
- 输入：直接使用Thinker输出的隐藏表示
- 输出：音频token序列
- 特点：确保语音生成与文本语义的一致性

#### 2.2.3 端到端设计优势

- Thinker和Talker**联合训练和推理**
- 避免了文本和语音两个模态之间的相互干扰
- 实现了真正的端到端多模态生成

### 2.3 TMRoPE（Time-aligned Multimodal RoPE）

为了同步视频输入和音频的时间戳，Qwen2.5-Omni提出了创新的位置编码方案 **TMRoPE**：

- 全称：Time-aligned Multimodal Rotary Position Embedding
- 核心思想：将音频和视频按照时间顺序**交错排列**
- 确保不同模态的时间信息在位置编码中对齐
- 使模型能够理解多模态信息的时序关系

### 2.4 流式处理架构

#### 2.4.1 块式处理（Block-wise Processing）

- 音频编码器和视觉编码器均采用**块式处理**方式
- 允许模型在接收输入的同时开始处理
- 实现多模态信息的实时流式输入

#### 2.4.2 滑动窗口DiT（Sliding-window DiT）

用于将音频token解码为实际波形：
- 采用**滑动窗口注意力机制**的扩散Transformer（DiT）
- 限制感受野范围
- 目的：**减少首包延迟**（initial package delay）
- 支持流式音频生成

### 2.5 多模态输入处理

| 模态 | 编码器 | 处理方式 |
|------|--------|---------|
| 文本 | 文本编码器（Tokenizer） | 直接编码为token序列 |
| 图像 | Qwen2.5-VL 视觉编码器 | 块式特征提取 |
| 视频 | Qwen2.5-VL 视觉编码器 | 逐帧块式处理 |
| 音频 | 音频编码器（Whisper系列） | 块式频谱特征提取 |

### 2.6 性能表现

| 对比维度 | Qwen2.5-Omni表现 |
|---------|------------------|
| 视觉理解 | 与同等规模的 Qwen2.5-VL 相当 |
| 音频理解 | 超越 Qwen2-Audio |
| 多模态综合 | Omni-Bench SOTA |
| 语音指令跟随 | 与纯文本输入性能相当（MMLU、GSM8K） |
| 语音生成 | 流式Talker超越大多数流式和非流式替代方案 |

**关键亮点**：端到端语音指令跟随能力与文本输入性能相当，表明模型真正实现了模态无关的理解能力。

### 2.7 语音生成能力

- 流式自然语音生成
- 鲁棒性和自然度优于大多数现有方案
- 支持实时对话场景
- 通过DiT解码器将离散音频token转换为连续波形

---

## Part 3：专业方向模型概述

### 3.1 Qwen2.5-Coder（代码生成）

#### 基本信息
- 论文：*Qwen2.5-Coder Technical Report* (arXiv: 2409.12186)
- 发布时间：2024年9月
- 前身：CodeQwen1.5
- 模型规模：0.5B / 1.5B / 3B / 7B / 14B / 32B

#### 核心特点
- 基于 Qwen2.5 架构，继续在超过 **5.5万亿token** 的代码语料上预训练
- 通过精细数据清洗、可扩展的合成数据生成、平衡数据混合实现卓越代码能力
- **保留通用语言和数学能力**的同时专精代码
- 许可证：Apache 2.0（大多数尺寸）

#### 能力覆盖
| 能力 | 描述 |
|------|------|
| 代码生成 | HumanEval、MBPP等基准SOTA |
| 代码补全 | 上下文感知的代码续写 |
| 代码推理 | 理解和分析代码逻辑 |
| 代码修复 | Bug定位和修复 |

#### 性能
- 在超过10个代码相关基准测试中达到SOTA
- 小模型（如7B）能够与更大通用模型竞争
- Qwen2.5-Coder-7B-Instruct在多种编程语言中超越许多更大模型

---

### 3.2 Qwen2.5-Math（数学推理）

#### 基本信息
- 论文：*Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement* (arXiv: 2409.12122)
- 发布时间：2024年9月
- 前身：Qwen2-Math
- 模型规模：1.5B / 7B / 72B

#### 核心特点
- 在大规模数学相关数据上预训练，包含 Qwen2-Math 生成的合成数据
- **支持中英双语**数学问题求解
- 支持多种推理方法：
  - **CoT**（Chain-of-Thought）：链式推理
  - **PoT**（Program-of-Thought）：程序化推理
  - **TIR**（Tool-Integrated Reasoning）：工具集成推理

#### 性能
- Qwen2.5-Math-72B-Instruct 超越 Qwen2-Math-72B-Instruct 和 GPT-4o
- 即使1.5B的小模型也能达到与大型语言模型高度竞争的数学推理能力
- MATH benchmark 80+

---

### 3.3 QwQ（推理模型）

#### 基本信息
- 首个预览版：QwQ-32B-Preview（2024年11月）
- 正式版：QwQ-32B（2025年3月）
- 参数量：32B
- 许可证：Apache 2.0

#### 核心理念

QwQ是Qwen系列的**推理专用模型**，通过大规模强化学习（RL）训练，实现深度思考和复杂推理能力。

#### 技术特点
- 基于强大的预训练基础模型 + **规模化强化学习**
- 不同于传统指令微调模型，QwQ能够进行思考和推理
- 集成了**Agent相关能力**：批判性思维 + 工具使用 + 基于环境反馈的推理调整
- 32B参数实现了与 DeepSeek-R1（671B参数，37B激活）相当的性能

#### 性能对比
- 数学推理：与 DeepSeek-R1、o1-mini 竞争
- 代码能力：强大的编程推理
- 通用问题解决：全面的推理能力
- 具体基准包括 AIME、MATH、LiveCodeBench 等

---

### 3.4 Qwen2.5-VL（视觉语言模型）

#### 基本信息
- 属于Qwen视觉-语言模型系列
- 规模：包括72B等多种尺寸
- 采用动态分辨率处理

#### 核心能力
- 图像理解与分析
- 视频理解
- 文档/表格理解
- 视觉定位（Grounding）
- Agent能力（GUI操作）

#### 后续发展
- **Qwen3-VL**（2025年）：视觉语言模型系列最强版本，全面升级

---

### 3.5 Qwen3 系列（2025年5月+）

#### Qwen3 LLM
- 发布时间：2025年5月
- 系列模型，涵盖多种规模
- 继续在语言理解、推理、代码等方面全面提升

#### Qwen3-Omni
- 在Qwen2.5-Omni的Thinker-Talker架构基础上进行五大关键升级
- 首个在文本、图像、音频和视频上同时保持SOTA的单一模型

#### Qwen3-VL
- 视觉语言模型系列最新最强版本
- 全面升级的视觉理解能力

---

## Part 4：Qwen模型生态全景图与时间线

### 4.1 模型族谱时间线

```
2023年
├── 08月  Qwen-7B（首个基座模型）
├── 08月  Qwen-VL（首个视觉语言模型）
├── 11月  Qwen-Audio（首个音频模型）
├── 12月  Qwen-72B

2024年
├── 03月  CodeQwen1.5
├── 06月  Qwen2 系列（0.5B~72B）
├── 06月  Qwen2-VL
├── 07月  Qwen2-Audio
├── 08月  Qwen2-Math
├── 09月  Qwen2.5 系列（0.5B~72B）
├── 09月  Qwen2.5-Coder（0.5B~32B）
├── 09月  Qwen2.5-Math（1.5B~72B）
├── 10月  Qwen2.5-VL
├── 11月  QwQ-32B-Preview（首个推理模型）

2025年
├── 03月  QwQ-32B（正式版推理模型）
├── 03月  Qwen2.5-Omni（首个全模态模型）
├── 05月  Qwen3 LLM 系列
├── 05月  Qwen3-VL / Qwen3-Omni
├── 07月  Qwen3-Coder 480B-A35B（旗舰代码模型）
├── 09月  Qwen3-Max（万亿参数，API-only）
├── 09月  Qwen3-Next-80B-A3B（混合注意力新架构）

2026年
├── 02月  Qwen3.5 (397B-A17B, 开源)
├── 02月  Qwen3.5-Plus (API)
├── 04月  Qwen3.5-Omni（全模态升级）
├── 05月  Qwen3.6 系列（开源新一代）
├── 05月  Qwen3.7-Max（Agent Frontier旗舰）
├── 05月  Qwen3.5-LiveTranslate-Flash（实时同声传译）
├── 05月  Qwen3-Coder-Next（80B-A3B，本地编码Agent）
├── 06月  Qwen3.7-Plus（多模态Agent基座）
```

### 4.2 各方向定位差异

| 方向 | 代表模型 | 核心定位 | 关键差异化 |
|------|---------|---------|------------|
| 通用语言 | Qwen3.5 / Qwen3.6 / Qwen3.7 | 通用知识与推理 | 36T+ token预训练，262K-1M上下文 |
| 视觉语言 | Qwen3-VL / Qwen3.7-Plus | 图像/视频理解 + Agent | 动态分辨率，视觉Agent，GUI操控 |
| 音频理解 | Qwen-Audio / Qwen2-Audio | 通用音频理解 | Whisper编码器，30+任务覆盖 |
| 全模态 | Qwen3.5-Omni | 统一感知+生成 | 数百亿参数，215基准SOTA |
| 代码 | Qwen3-Coder / Qwen3-Coder-Next | 代码智能与Agent | 7.5T代码token，本地/云端Agent |
| 数学 | Qwen2.5-Math | 数学推理 | CoT+PoT+TIR，中英双语 |
| 推理 | QwQ | 深度推理 | 强化学习训练，长链推理 |
| 同声传译 | Qwen3.5-LiveTranslate | 实时翻译 | 60语言，2.8秒延迟，声音克隆 |

### 4.3 技术演进脉络

#### 音频方向的演进：
1. **Qwen-Audio**（2023）：层次化标签 → 统一多任务预训练 → 30+任务覆盖
2. **Qwen2-Audio**（2024）：自然语言提示替代标签 → DPO对齐 → 双模式交互
3. **Qwen2.5-Omni**（2025）：全模态统一 → Thinker-Talker → 流式生成
4. **Qwen3.5-Omni**（2026）：数百亿参数 → 215基准SOTA → 可控音视频字幕
5. **Qwen3.5-LiveTranslate**（2026）：实时同声传译 → 60语言 → 声音克隆 → 2.8秒延迟

#### 架构演进的核心趋势：
- **编码器升级**：Whisper-large-v2 → Whisper-large-v3 → 块式处理 → 原生多模态融合
- **训练范式**：多任务标签 → 自然语言提示 → 端到端联合训练 → 大规模RL对齐
- **对齐技术**：SFT → SFT + DPO → GRPO + R1-Zero
- **模态整合**：单模态 → 双模态 → 全模态统一 → **多模态Agent统一**
- **设计理念**：感知理解 → 生成能力 → Agent原生（看、想、写、做、验）

### 4.4 开源策略

Qwen系列模型采用积极的开源策略：
- 绝大多数模型采用 **Apache 2.0** 许可证
- 提供完整的模型权重、代码和推理脚本
- 支持主流框架：Hugging Face Transformers、vLLM、Ollama、llama.cpp等
- 提供商业API服务：Qwen-Plus、Qwen-Turbo、Qwen3.7-Max等
- 2026年1月HuggingFace累计下载量突破7亿
- 开源+闭源双轨策略：开源驱动生态，闭源提供前沿能力

---

## Part 5：2026年最新进展补充

### 5.1 Qwen3.5-Omni（2026年4月）

Qwen3.5-Omni是Qwen-Omni系列的最新版本，相比Qwen3-Omni实现三大新能力：

- **可控的音视频字幕生成**：支持精细控制的字幕生成
- **百亿级参数规模**：音频视觉理解和生成能力均达到SOTA
- **旗舰版Qwen3.5-Omni-Plus**：在215个音频/音视频基准中达到SOTA，超趇Gemini 3.1 Pro
- **技术报告**：arXiv:2604.15804
- **视觉编码器**：采用Qwen3.5的视觉编码器处理图像和视频

### 5.2 Qwen3.5-LiveTranslate-Flash（2026年5月19日）

基于Qwen3.5-Omni构建的实时同声传译模型：

| 特征 | 详情 |
|------|------|
| 支持语言 | 60种（29种语音 + 31种文本） |
| 端到端延迟 | 2.8秒 |
| 声音克隆 | 自动复制说话者声音特征 |
| 视觉增强 | 支持多模态实时翻译 |
| 场景 | 会议、直播、视频通话等实时场景 |

### 5.3 Qwen3-Coder系列更新

- **Qwen3-Coder 480B-A35B**（2025年7月）：480B总参数/35B激活，7.5T tokens训练（70%代码），1M上下文（YaRN），SWE-Bench Verified SOTA，匹配Claude Sonnet 4
- **Qwen3-Coder-Next**（2026年5月）：80B总参数/3B激活，混合注意力架构（Gated DeltaNet），256K上下文，专为本地编码Agent设计，达到Claude Sonnet 4.5级编码性能

### 5.4 团队动态

- 2026年1月：Qwen模型在HuggingFace累计下载量突破7亿
- 2026年3月：技术负责人林俊杨离职（曾主导Qwen3-Max开发）
- 2026年5月：团队由Steven Hoi领导，定位"Foundation Models for the Agent Era"

---

## 论文引用列表

### 音频/全模态方向

1. **Qwen-Audio**  
   Chu, Y., Xu, J., Zhou, X., Yang, Q., Zhang, S., Yan, Z., Zhou, C., & Zhou, J. (2023). Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models. *arXiv preprint arXiv:2311.07919*.

2. **Qwen2-Audio**  
   Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng, Y., Lv, Y., He, J., Lin, J., Zhou, C., & Zhou, J. (2024). Qwen2-Audio Technical Report. *arXiv preprint arXiv:2407.10759*.

3. **Qwen2.5-Omni**  
   Xu, J., Guo, Z., He, J., Hu, H., He, T., Bai, S., Chen, K., Wang, J., Fan, Y., Dang, K., Zhang, B., Wang, X., Chu, Y., & Lin, J. (2025). Qwen2.5-Omni Technical Report. *arXiv preprint arXiv:2503.20215*.

### 通用/专业方向

4. **Qwen2.5**  
   Qwen Team. (2024). Qwen2.5 Technical Report. *arXiv preprint arXiv:2412.15115*.

5. **Qwen2**  
   Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., ... & others. (2024). Qwen2 Technical Report. *arXiv preprint arXiv:2407.10671*.

6. **Qwen2.5-Coder**  
   Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., ... & Lin, J. (2024). Qwen2.5-Coder Technical Report. *arXiv preprint arXiv:2409.12186*.

7. **Qwen2.5-Math**  
   Qwen Team. (2024). Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement. *arXiv preprint arXiv:2409.12122*.

8. **QwQ-32B**  
   Qwen Team. (2025). QwQ-32B: Embracing the Power of Reinforcement Learning. *Qwen Blog*.

9. **Qwen-VL**  
   Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., & Zhou, J. (2023). Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities. *arXiv preprint arXiv:2308.12966*.

### 关键技术依赖

10. **Whisper**  
    Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML 2023*.

---

*报告完*
