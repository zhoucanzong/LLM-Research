# GLM 系列模型深度调研报告

> 调研对象：智谱 AI（Zhipu AI / Z.ai）与清华大学 KEG 实验室（THUDM）联合研发的 GLM/ChatGLM/CogVLM/CogVideoX/GLM-Z1/GLM-4.5 等模型家族
> 时间跨度：2021 年 CogView 至 2025–2026 年 GLM-4.5/4.6/4.7
> 调研日期：2026-06-04
> 资料来源：arXiv 论文、Z.ai 官方博客与文档（z.ai / bigmodel.cn）、GitHub（THUDM、zai-org）、Hugging Face、ICLR/NeurIPS/ACL 等会议论文

---

## 一、概述

智谱 AI（前身：清华大学 KEG 实验室孵化的中国 AI 独角兽）是国内最早系统性自研基础大模型并开源的机构之一。其模型家族覆盖：

- **基座语言模型**：GLM (2021/22) → GLM-130B (2022) → ChatGLM-6B / ChatGLM2-6B / ChatGLM3-6B (2023) → GLM-4 / GLM-4-9B / GLM-4-32B-0414 (2024–2025) → GLM-4.5 / GLM-4.5-Air / GLM-4.6 / GLM-4.7 (2025–2026)
- **推理（深度思考）模型**：GLM-Z1-9B/32B-0414、GLM-Z1-Rumination-32B、GLM-4.5 Thinking Mode (2025)
- **视觉语言模型 (VLM)**：CogVLM-17B (2023) → CogVLM2 / CogVLM2-Video (2024) → GLM-4V-9B (2024) → GLM-4.1V-Thinking (2025) → GLM-4.5V / GLM-4.6V (2025)
- **图像生成**：CogView (NeurIPS 2021) → CogView2 (NeurIPS 2022) → CogView3 / CogView-3-Plus (2024) → CogView4 (2025)
- **视频生成**：CogVideo (2022) → CogVideoX-2B/5B (ICLR 2025)

GLM 系列的独特之处在于：早期（2021–2023）坚持自主路线 —— 采用 **Prefix LM + 自回归填空 (Autoregressive Blank Infilling)** 的混合预训练目标，与 GPT 风格 Causal LM 截然不同；从 ChatGLM2 开始逐步收敛到主流 Causal LM 范式（RoPE + RMSNorm + SwiGLU + GQA），并在 GLM-4.5 转向 MoE 与 Hybrid Reasoning 路线。本报告以"架构演进 + 关键创新"双线索梳理这一脉络。

### 1.1 时间线总览

| 时间 | 模型 | 关键特征 |
|---|---|---|
| 2021-05 | CogView | 4B 参数，Transformer + dVAE，文本→图像生成（NeurIPS 2021） |
| 2021-12 | GLM (论文 v1) | 提出"自回归填空"预训练目标；2D 位置编码；ACL 2022 |
| 2022-04 | CogView2 | 层次化生成 + Cross-modal General LM，10× 加速 |
| 2022-05 | CogVideo | 9B，首个开源大规模文生视频模型，多帧生成 |
| 2022-08 | GLM-130B | 1300 亿参数双语模型；INT4 量化首次在百亿级实现（ICLR 2023） |
| 2023-03 | ChatGLM-6B | 首个对外开源的中文对话模型，6.2B，GLM 架构 + SFT/RLHF |
| 2023-06 | ChatGLM2-6B | 切换到 GLM 混合目标 + FlashAttention + Multi-Query Attention，32K 上下文 |
| 2023-10 | ChatGLM3-6B | 3D-RoPE / MQA / Causal-only mask；支持工具调用与代码解释器 |
| 2023-11 | CogVLM-17B | "Visual Expert"机制：在每层 Attention/FFN 中并行视觉专家 |
| 2024-01 | GLM-4 (闭源旗舰) | 对标 GPT-4；GLM-4 All Tools |
| 2024-05 | CogVLM2 / GLM-4V-9B | 切换到 LLaMA3-8B 底座；图像 1344×1344；19B 总参 |
| 2024-06 | GLM-4-9B 开源 + ChatGLM 综述论文 (arXiv 2406.12793) | 10T tokens 预训练 |
| 2024-08 | CogVideoX (2B/5B) | 3D Causal VAE + Expert Transformer + Expert AdaLN |
| 2024-09 | CogView3 / CogView-3-Plus | Relay Diffusion（级联扩散），DiT 路线 |
| 2025-03 | CogView4-6B | 进一步扩展到中文原生 DiT |
| 2025-04 | GLM-4-32B-0414 + GLM-Z1-32B + GLM-Z1-Rumination-32B | 首个开源旗舰级稠密模型 + 深度思考 + 反刍模型 |
| 2025-07 | GLM-4.1V-9B-Thinking | VLM 引入 RL 训练的"Thinking"链 |
| 2025-07 | **GLM-4.5 / GLM-4.5-Air** | MoE 355B/A32B 与 106B/A12B；Hybrid Reasoning；slime RL 框架 |
| 2025-08 | GLM-4.5V | 基于 GLM-4.5-Air 的 106B MoE VLM |
| 2025-09 | GLM-4.6 | 357B MoE，200K 上下文，Agent/Coding 强化 |
| 2025-12 | GLM-4.7 | "Preserved Thinking"，专注自主编程 |
| 2025-12 | GLM-4.6V-Flash | 9B 端侧 VLM |

> 注：智谱 2025 年起将海外品牌名启用 **Z.ai**，开源仓库迁至 **github.com/zai-org**（原 THUDM 仓库部分仍保留）。

---

## 二、基座语言模型演进

### 2.1 GLM（General Language Model，2021–2022）

**核心论文**：Du et al., *GLM: General Language Model Pretraining with Autoregressive Blank Infilling*, ACL 2022（arXiv:2103.10360）。

**动机**：当时主流预训练范式分为三类——自编码（BERT，擅长 NLU）、自回归（GPT，擅长生成）、Encoder-Decoder（T5/BART，擅长 Seq2Seq）。GLM 旨在用一个统一架构同时适配三类下游任务。

**核心设计**：

1. **自回归填空 (Autoregressive Blank Infilling)**：从输入序列中随机采样若干 span 用 `[MASK]` 替换，剩余部分构成 Part A（双向可见），被 mask 的 spans 拼成 Part B（自回归生成）。模型在 Part A 上做双向编码，在 Part B 上做自回归解码——这本质上是一种 **Prefix LM**（Part A 全可见 + Part B 因果可见）。
2. **2D Positional Encoding**：每个 token 用 (position1, position2) 双坐标编码 —— position1 是其在原文中的绝对位置（被 mask 的 span 共享同一个位置 id），position2 是 span 内部的相对位置。这一设计使被填充内容的长度对模型不可见（避免信息泄漏）。
3. **Span 顺序随机化**：训练时打乱 spans 的预测顺序，强迫模型学习双向依赖。
4. **多任务预训练**：短 span（句中 mask，类 BERT）+ 长 span（句尾 mask，类 GPT）联合训练，使同一参数体可在 NLU、有条件/无条件生成任务上微调。

**模型规模**：GLM-Base (110M)、GLM-Large (335M)、GLM-XL (10B)、GLM-XXL (515M+)。

**架构细节**：基于 Transformer，但相比原始 BERT 做了两点改动：(a) 重排 LayerNorm 与残差连接的顺序（Post-LN→Pre-LN 思路）；(b) 输出 token 预测使用单层线性层；(c) 用 GeLU 替换 ReLU。

### 2.2 GLM-130B（2022-08，ICLR 2023）

**核心论文**：Zeng et al., *GLM-130B: An Open Bilingual Pre-Trained Model*, ICLR 2023（arXiv:2210.02414）。

**目标**：训练一个达到 GPT-3 (davinci, 175B) 水平的双语（中英）开源百亿级模型，并公开训练日志。

**架构选择**（与 GLM-Base 相同的预训练目标，但工程上做了多项关键决定）：

| 设计点 | GLM-130B 选择 | 备注 |
|---|---|---|
| 主架构 | GLM 自回归填空（Prefix LM） | 沿用 GLM 论文 |
| 层数 / 隐藏维度 | 70 层 / 12,288 hidden / 96 heads | 130B 参数 |
| 上下文长度 | 2,048 | 当时主流 |
| 位置编码 | **RoPE (Rotary Positional Embedding)** | 已抛弃 GLM-Base 的 2D learnable PE |
| 激活函数 | **GeGLU** | GLU 家族 |
| LayerNorm | **DeepNorm**（自研） | 解决 100B 级训练发散 |
| 训练精度 | FP16 主体 + 梯度累加 FP32 | FlashAttention 之前的方案 |
| 推理量化 | **INT4**（无后训练） | 首个达到 INT4 几乎无损的 100B+ 模型，可在 4×RTX 3090 上推理 |

**训练**：在 96×A100 集群上耗时 60 天，预训练 400B tokens（200B 中文 + 200B 英文）。论文重点报告了训练稳定性（loss spike 与 divergence）的工程经验：嵌入梯度收缩 (Embedding Gradient Shrink, EGS)、混合精度策略调整、DeepNorm 替换 Pre/Post-LN 等。

**评测**：在 MMLU、LAMBADA、BIG-bench-Lite 上接近或超过 GPT-3 davinci；在中文 CLUE/FewCLUE 上显著超过 ERNIE-Titan-3.0 260B。

> **重要**：GLM-130B 是该家族**仍然采用 Prefix LM 双向+单向混合**的最后一代百亿级模型。下一代 ChatGLM 系列在工程上继续兼容 GLM 目标，但从 ChatGLM3 开始已逐步向标准 Causal LM 收敛。

### 2.3 ChatGLM-6B / ChatGLM2-6B / ChatGLM3-6B（2023）

| 维度 | ChatGLM-6B (2023.03) | ChatGLM2-6B (2023.06) | ChatGLM3-6B (2023.10) |
|---|---|---|---|
| 参数 | 6.2B（28 GLM Block） | 6.2B | 6.2B |
| 预训练目标 | **GLM 自回归填空（Prefix LM）** | GLM 混合目标，1.4T 双语 tokens | 与 ChatGLM2 类似，更大规模数据 |
| 位置编码 | RoPE | RoPE（单维改造） | RoPE |
| Attention | 标准 MHA | **Multi-Query Attention (MQA)** | MQA |
| 激活 | GeLU | **SwiGLU**（部分版本仍 GeLU；社区资料口径不一） | SwiGLU |
| LayerNorm | RMSNorm | RMSNorm | RMSNorm |
| 上下文 | 2K | **32K**（FlashAttention） | **128K**（ChatGLM3-6B-128K） |
| 对齐 | SFT + RLHF（小规模） | SFT + Human Preference Alignment | SFT + DPO 风格对齐 |
| 工具能力 | — | — | **原生 Function Call / Code Interpreter / Agent** Prompt |
| 开源 | 首发开源，引爆国内开源浪潮 | 推理速度提升 42%，KV cache 显存降至 1/8 | 兼容三种 prompt：chat、tool、code |

**关键演进**：

1. ChatGLM2 引入 **MQA**（所有 head 共享同一组 K/V，仅 Query 多头）大幅压缩 KV cache，显存占用降为 MHA 的约 1/N；为后来 GQA 在国产大模型中的普及打下工程基础。
2. ChatGLM2 启用 **FlashAttention**（v1）+ 上下文从 2K→32K。
3. ChatGLM3 重点布局**工具能力（Tools）**：模板支持 system / user / assistant / observation / tool 多角色，可被视为后续 GLM-4 All Tools 的前身。
4. 三代 ChatGLM-6B 在 Hugging Face 累计下载量在 2023 年突破千万，是中文开源生态的标志性模型。

### 2.4 GLM-4（2024）

**官方综述**：Team GLM, *ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools*, arXiv:2406.12793 (2024-06)。

**模型矩阵**：

| 模型 | 类型 | 上下文 | 备注 |
|---|---|---|---|
| GLM-4 | API 旗舰（闭源） | 128K | 对标 GPT-4 |
| GLM-4-Air | API 轻量 | 128K | 高性价比 |
| GLM-4-9B | 开源 base | 8K | 2024-06 开源 |
| GLM-4-9B-Chat | 开源 chat | 128K / 1M | 长上下文版本 |
| GLM-4V-9B | 开源 VLM | 8K | 见第三章 |

**训练规模**：GLM-4 系列预训练 **10T tokens**（中英为主，少量 24 语种），多阶段后训练（SFT + 来自人类反馈的强化学习/偏好对齐）。

**评测**（论文摘要）：
- **MMLU / GSM8K / MATH / BBH / GPQA / HumanEval**：与 GPT-4 持平或略优；
- **IFEval（指令跟随）**：接近 GPT-4-Turbo；
- **AlignBench（中文对齐）**：超过 GPT-4。

**GLM-4 All Tools**：模型自主决定何时调用 web browser / Python interpreter / Text-to-Image / 用户自定义函数，在线访问与数学求解上匹敌 GPT-4 All Tools。这是国内最早系统化 Function-Calling Agent 化的旗舰模型之一。

### 2.5 GLM-4-32B-0414 与 GLM-Z1（2025-04）

2025-04-14 智谱发布 **GLM-4-0414 系列**，将开源主线从 9B 跳到 32B，并首次系统性开源推理模型。

| 模型 | 类型 | 上下文（YaRN 扩展） | 说明 |
|---|---|---|---|
| GLM-4-32B-Base-0414 | Base | 32K → 128K | 32B 稠密底座 |
| GLM-4-32B-0414 | Chat | 32K → 128K | 通用对话/Agent |
| GLM-Z1-32B-0414 | Reasoning | 32K → 128K | "深度思考"模型 |
| GLM-Z1-Rumination-32B-0414 | Deep Research | 128K | "反刍/沉思"模型，对标 OpenAI Deep Research |
| GLM-4-9B-0414 | Chat | 32K → 128K | 轻量批量任务 |
| GLM-Z1-9B-0414 | Reasoning | 32K → 128K | 9B 推理"惊喜版" |

特性：
- **YaRN RoPE Scaling**：当输入 + 输出超过 32K 时启用 `factor=4.0` 的 YaRN，扩展到 128K；
- **GLM-4-32B-0414 函数调用 / Agent**：在 BFCL-v3 (69.6)、TAU-Bench Retail (68.7)、Airline (51.2)、IFEval (87.6) 等指标上反超 DeepSeek-V3-0324 与 GPT-4o-1120；
- **SWE-bench Verified**：搭配 Moatless 框架达到 33.8（同等规模开源 SOTA）；
- 推理速度（在线）最高达 **200 tokens/s**。

详见第五章"推理模型"。

### 2.6 GLM-4.5 / GLM-4.5-Air（2025-07）—— MoE 与 Hybrid Reasoning

**核心论文**：*GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*, arXiv:2508.06471 (2025-08)。

**架构关键参数**：

| 模型 | 总参数 | 激活参数 | 上下文 | 最大输出 |
|---|---|---|---|---|
| GLM-4.5 | 355B | 32B | 128K | 96K |
| GLM-4.5-Air | 106B | 12B | 128K | 96K |
| GLM-4.5-X / AirX / Flash | API 变体（速度/价格不同档） | — | 128K | 96K |

**架构设计要点**（与 DeepSeek-MoE 同源、有所改造）：
1. **Mixture-of-Experts**：每层多个 FFN 专家，每 token 动态路由到 Top-K；
2. **Loss-free Balance Routing**：替代经典的 auxiliary load-balancing loss，避免对主任务 loss 的负面干扰；
3. **Sigmoid 路由 + 共享专家**（部分细节与 DeepSeek 相似）；
4. **Hybrid Reasoning**：模型同时支持 Thinking Mode（带显式 reasoning_content）与 Non-Thinking Mode，通过 API 参数 `thinking.type = enabled / disabled` 切换；
5. **训练管线**：15T tokens 通用预训练 → 代码/推理/Agent 任务定向后训练 → RL（slime 框架）增强；技术报告中的总训练 token 数为 **23T**（含后训练 RL rollout）。

**slime（强化学习框架）**：智谱自研、开源（github.com/THUDM/slime），是 GLM-4.5/4.6/4.7/5/5.1 系列共同的 RL 后训练基础设施。核心特性：
- **Hybrid 同步/异步训练架构**：支持 sync 与 fully-asynchronous decoupled rollouts；
- 把 RL rollout 视为"在线数据集生成器"，rollout 与 trainer 解耦扩缩容；
- 支持百亿级 MoE 的大规模 RL。

**性能**（来自官方与第三方报告）：
- 在 **MMLU Pro / AIME24 / MATH 500 / SciCode / GPQA / HLE / LiveCodeBench / SWE-Bench Verified / Terminal-bench / TAU-Bench / BFCL-v3 / BrowseComp** 12 项基准的综合平均分，**全球排名第 2、开源第 1**；
- GLM-4.5-Air（106B/A12B）在多项推理基准上**超越 Gemini 2.5 Flash、Qwen3-235B、Claude 4 Opus**；
- API 价格：输入 \$0.2 / M tokens、输出 \$1.1 / M tokens；高速版生成速度 100+ tokens/s。

### 2.7 GLM-4.6（2025-09）

**架构**：MoE，**357B 总参 / 32B 激活**（与 GLM-4.5 同代但参数微调），**上下文 200K**（GLM-4.5 为 128K），最大输出 131K。

**重点改进**（来自 Hugging Face zai-org/GLM-4.6 与第三方评测）：
- 推理性能显著提升，并支持**推理过程中调用工具**（thinking + tool use 交错）；
- Coding（特别是 Agentic Coding）相对 GLM-4.5 大幅提升；
- 在 8 项公开 Agent / Reasoning / Coding 基准上系统性优于 GLM-4.5；
- 与 Claude Sonnet 4.5 在编码与推理上"近平价"，但成本显著更低。

### 2.8 GLM-4.7 与 GLM-5（2025-12 起）

- **GLM-4.7**（2025-12）：定位**自主编程**，引入 *Preserved Thinking* —— 在多轮工具调用之间保留思考链，避免 reasoning 被截断；
- **GLM-5 / GLM-5.1**（2025 末–2026 初）：进一步扩展，bigmodel.cn 主推作为新一代旗舰；继续基于 slime RL 框架训练；
- 社区流出的部分版本提到**744B MoE / 40B 激活**配置，但官方未在论文中正式确认，本报告标注为非权威信息。

---

## 三、视觉语言模型 (VLM) 演进

### 3.1 CogVLM（2023-11）—— Visual Expert 机制

**核心论文**：Wang et al., *CogVLM: Visual Expert for Pretrained Language Models*, NeurIPS 2024（arXiv:2311.03079）。

**核心动机**：当时主流多模态范式（BLIP-2、LLaVA、MiniGPT-4）多采用 **shallow alignment**——把图像特征通过 Q-Former / 简单 MLP 投到 LLM 的输入 embedding 空间，LLM 主体冻结。CogVLM 团队认为这种"浅对齐"会损害视觉细节（OCR、grounding、细粒度物体）。

**架构**（CogVLM-17B = ViT 4.4B + LLM 7B + Visual Expert ~5B）：

```
                ┌─────────────────────────────┐
                │      ViT (EVA-CLIP-E/14)    │
                └──────────────┬──────────────┘
                               │ patch tokens
                ┌──────────────▼──────────────┐
                │  MLP Adapter (2-layer)       │
                └──────────────┬──────────────┘
                               │ visual tokens (concat with text)
   ┌───────────────────────────▼───────────────────────────┐
   │  每一层 Transformer Block (Vicuna-7B 主干)              │
   │  ┌───────────────────────────────────────────────┐    │
   │  │   Self-Attention                              │    │
   │  │   Q/K/V = [Q_text/K_text/V_text  ‖            │    │
   │  │            Q_visual/K_visual/V_visual]        │    │
   │  │   注意力矩阵跨 modality 共享                    │    │
   │  └───────────────────────────────────────────────┘    │
   │  ┌───────────────────────────────────────────────┐    │
   │  │   FFN_text  (frozen LLM expert)               │    │
   │  │   FFN_visual (Visual Expert，trainable)        │    │
   │  │   token 是 visual 走 FFN_visual，否则 FFN_text  │    │
   │  └───────────────────────────────────────────────┘    │
   └────────────────────────────────────────────────────────┘
```

**关键点**：
1. **每一层都注入"视觉专家" QKV+FFN**，参数数量约等于该层原本的 QKV+FFN，相当于在每层并行一个"视觉副本"；
2. **token 路由**：text token 走 LLM 原参数，visual token 走视觉专家；attention 层跨模态共享 Q/K/V 计算（拼接后 softmax）；
3. **冻结 LLM 主参数**，只训练 Visual Expert + ViT + MLP adapter，**完全不损害原 LLM 的纯文本性能**（这是与 LLaVA 等"开放 LLM 训练"路线的最大区别）；
4. **深度融合（Deep Fusion）**：与浅对齐相比，视觉信号在每一层都参与 attention，提升 OCR / grounding / 细粒度推理。

**评测**：CogVLM-17B 在 NoCaps、Flickr30k captioning、RefCOCO/+/g、Visual7W、GQA、ScienceQA、VizWiz VQA、TDIUC 共 **10 个跨模态基准上达到 SOTA**，并在 VQAv2、OKVQA、TextVQA、COCO captioning 上仅次于 PaLI-X 55B（参数量约为 1/3）。

### 3.2 CogVLM2 / CogVLM2-Video / GLM-4V-9B（2024-05 / 08）

**核心论文**：Hong et al., *CogVLM2: Visual Language Models for Image and Video Understanding*, arXiv:2408.16500 (2024-08)。

CogVLM2 家族继续采用 **Visual Expert 架构**，但做了多项工程升级：

| 维度 | CogVLM-17B | CogVLM2-19B (llama3-chat-19B) | GLM-4V-9B |
|---|---|---|---|
| LLM 底座 | Vicuna-7B | **Meta-Llama-3-8B-Instruct** | **GLM-4-9B** |
| 视觉编码器 | EVA-CLIP-E/14 | **SigLIP / EVA-CLIP** 改造版 | 同 CogVLM2 |
| MLP Adapter | 2 层 MLP | **2×2 卷积 + SwiGLU** | 同 CogVLM2 |
| 图像分辨率 | 490² | **1344×1344** | 1120/1344 |
| 上下文 | 2K | **8K** | 8K |
| 总参 | 17B | 19B（8B LLM + 11B 视觉专家） | 13B 级 |
| 中文 | — | cogvlm2-llama3-chinese-chat-19B | 原生中英 |

**CogVLM2-Video**（2024-07）：
- 在 CogVLM2 基础上增加**时间 grounding** 数据构建管线，自动生成 30K 时间相关 QA；
- 支持分钟级视频理解；
- 在多项 video QA 基准上 SOTA（同期开源）。

**GLM-4V-9B**：作为 GLM-4 主线的视觉版本，继承 CogVLM2 的视觉专家结构，绑定 GLM-4-9B 文本基座，是 ChatGLM 综述论文中正式列入的开源 VLM。

### 3.3 GLM-4.1V-Thinking / GLM-4.5V / GLM-4.6V（2025）

**核心论文**：*GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with a Family of VLMs*, arXiv:2507.01006 (2025-07，多次修订至 v6 含 GLM-4.6V)。

**统一架构**（三代共享）：

```
Vision Encoder  →  MLP Projector  →  LLM (GLM-4-9B / GLM-4.5-Air / GLM-4.6 base)
```

变化要点：
1. **抛弃 Visual Expert**：从 GLM-4.1V 开始回归到主流 **vision encoder + projector + LLM** 结构，更易复用 LLM 基座的 RL 后训练成果；
2. **强化学习"Thinking"链**：大量使用强化学习 + curriculum 教模型在视觉推理上"显式思考"；GLM-4.1V-9B-Thinking 用 RL 让 9B 模型在复杂多模态推理基准上击败 72B 量级闭源模型；
3. **GLM-4.5V (2025-08)**：基于 **GLM-4.5-Air (106B/A12B)** 底座，继承 4.1V 思路；在 **42 项公开视觉-语言基准上达到同规模 SOTA**；支持图像、视频、长文档与 GUI Agent；
4. **GLM-4.6V**（2025 Q4）：升级到 GLM-4.6 底座，进一步扩展多模态推理与长视频；
5. **GLM-4.6V-Flash 9B**（2025-12）：端侧/低成本版本。

---

## 四、视觉生成模型

### 4.1 CogView 系列（图像生成）

| 模型 | 时间 | 关键技术 | 参数 |
|---|---|---|---|
| **CogView** | 2021-05 (NeurIPS 2021) | dVAE 离散 token + 4B GPT-style Transformer 自回归生成；首个开源中文文生图大模型 | 4B |
| **CogView2** | 2022-04 (NeurIPS 2022) | **Cross-modal General LM (CogLM)**：多任务 + 层次化生成；先生成低分辨率图像 token，再用 super-resolution 模块迭代细化；相对 DALL·E-2 提供 ~10× 生成加速 | 6B |
| **CogView3 / CogView-3-Plus** | 2024-03 (arXiv:2403.05121) | 转向**扩散模型**路线；提出 **Relay Diffusion** 级联框架——低分辨率 latent 扩散 + 高分辨率 relay refiner | 3B |
| **CogView4-6B** | 2025-03 | 中文原生 **DiT (Diffusion Transformer)**，宽高 512–2048（32 整除），BF16 | 6B |

CogView 路线显著的"风格"是：从早期 GPT-style 自回归逐步**迁移到 DiT/扩散**，与同期 DALL·E、Stable Diffusion、Imagen 演进路径相似，但保持与 GLM 主家族的工程协同（共享 tokenizer、训练框架）。

### 4.2 CogVideoX（视频生成）

**核心论文**：Yang et al., *CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer*, ICLR 2025（arXiv:2408.06072）。

**预演**：CogVideo (2022, ICLR 2023) 是首个开源 9B 文生视频模型，基于 CogView2 扩展时间维度，但生成时长与运动质量有限。

**CogVideoX 关键创新**：

1. **3D Causal VAE**：沿空间和时间两个维度对视频压缩，时间上采用 **causal**（仅看历史帧）以支持任意长度；显著降低 token 数量同时保持高保真重建；2024-08 单独开源。
2. **Expert Transformer + Expert Adaptive LayerNorm (Expert AdaLN)**：
   - 视频 latent token 与文本 token 拼接进入同一 Transformer，但 LayerNorm 的 affine 参数 (γ, β) 对**两种模态使用不同的 expert**；
   - 这种"按模态分通道、共享 self-attention"的范式，与 Stable Diffusion 3 的 MMDiT、与 CogVLM 的 Visual Expert 思路一脉相承——**深度融合不浅化任一模态**；
3. **Progressive Training + Multi-Resolution Frame Pack**：训练时混合不同分辨率/时长视频，用 frame pack 高效填充 batch；
4. **数据管线**：自研视频字幕模型 + 多阶段过滤；与字幕模型和 3D VAE 一同开源；
5. **生成能力**：10 秒、16 fps、768×1360 分辨率，连续叙事，运动幅度大。

**开源版本**：CogVideoX-2B（基础）/ CogVideoX-5B / CogVideoX-5B-I2V（图生视频）/ CogVideoX-1.5。在国产文生视频开源生态（与 Wan / HunyuanVideo / Open-Sora 并列）中是最早达到工业级长视频质量的模型之一。

---

## 五、推理模型（Reasoning / Deep Thinking）

### 5.1 GLM-Z1-32B-0414

- 基于 GLM-4-32B-0414 通过 **冷启动 + 大规模强化学习** 微调；
- 训练任务包括数学、代码、逻辑（含 AIME、MATH、GSM8K、LiveCodeBench、Codeforces 类问题）；
- 引入**基于 pairwise ranking 反馈的通用 RL**，避免推理能力提升以损失通用对话能力为代价；
- 在线推理速度 200 tokens/s；
- 在 AIME / MATH / GPQA / LiveCodeBench 上接近或超越同期 DeepSeek-R1-Distill-Llama-70B 等模型。

### 5.2 GLM-Z1-Rumination-32B-0414（"反刍模型"）

定位为对标 OpenAI **Deep Research**：
- "反刍 (Rumination)"：相对普通"思考"，可进行更长、更开放、更循环的搜索-思考-搜索；
- 训练通过端到端 RL 扩展，奖励信号来自 **ground-truth 答案 / rubrics**；
- 模型在思考过程中调用外部工具：`search` / `click` / `open` / `finish`（固定四工具集，外接搜索引擎）；
- 显著提升研究型写作（如"分析两座城市 AI 发展并撰写比较报告"）。

### 5.3 GLM-Z1-9B-0414 与 GLM-4.5 Hybrid Reasoning

- **GLM-Z1-9B**：用同套 RL 技术训练 9B 小模型，在数学推理与通用任务上是同尺寸开源模型 Top；
- **GLM-4.5 Thinking Mode**：把推理能力作为 MoE 旗舰内置模式，以 `thinking.type` 显式开关；这是从"独立推理模型"到"统一基座 + 推理模式"的范式转变（2024-09 OpenAI o1 之后业界主流方向，GLM-4.5 是开源端较早实现 hybrid reasoning 的 MoE 旗舰）。

---

## 六、跨代技术演进分析（重点：从 Prefix LM 到 Causal LM 的架构转型）

GLM 家族最具特色的演进，是从最初坚持"自回归填空"独特路线，逐步在保留品牌名"GLM"的同时，**事实上转向标准 Decoder-only Causal LM**。这一过程并非一次性切换，而是分阶段渐进：

### 6.1 关键演进节点

| 阶段 | 时间 | 代表模型 | 预训练目标 | Attention Mask | 位置编码 | Norm | 激活 |
|---|---|---|---|---|---|---|---|
| Prefix LM 时代 | 2021–2022 | GLM-Base/Large/XL/XXL | 自回归填空 (Part A 双向 + Part B 因果) | 混合 mask（Part A 全可见，Part B 因果） | **2D 可学习 PE** | Pre-LN | GeLU |
| 工程化双语 | 2022 | GLM-130B | 自回归填空 | 混合 mask | **RoPE** | **DeepNorm** | **GeGLU** |
| 对话化过渡 | 2023.03 | ChatGLM-6B | GLM 自回归填空 | 混合 mask | RoPE | RMSNorm | GeLU |
| MQA + 长上下文 | 2023.06 | ChatGLM2-6B | GLM 混合目标（同时含 BERT 风格 / GPT 风格 span） | 混合 mask（部分版本简化） | RoPE | RMSNorm | SwiGLU（部分） |
| 实质 Causal | 2023.10 | ChatGLM3-6B | 类 Causal LM（逐步弱化 blank infilling） | **Causal-only**（社区分析；官方文档不再强调 Prefix LM） | RoPE | RMSNorm | SwiGLU |
| 标准 Causal LM | 2024 | GLM-4 / GLM-4-9B | Causal LM | Causal | RoPE + YaRN 扩展 | RMSNorm | SwiGLU |
| 稠密旗舰 | 2025.04 | GLM-4-32B-0414 | Causal LM | Causal | RoPE + YaRN | RMSNorm | SwiGLU |
| MoE Hybrid Reasoning | 2025.07+ | GLM-4.5 / 4.6 / 4.7 | Causal LM + MoE + RL | Causal | RoPE | RMSNorm | SwiGLU + MoE FFN |

### 6.2 转型动因

1. **Causal LM 的"All-in-One"成熟**：GPT-3/4 用纯 Causal LM 同时把 NLU、生成、推理统一；T5/BART 这类编码-解码范式逐渐让位。Prefix LM 在工程上开发链路更复杂（需要为 Part A/B 维护混合 mask），而其相对 Causal LM 的下游优势在 SFT/RLHF 时代被逐步抹平。
2. **生态兼容**：迁移到标准 Causal LM 后可直接复用 Hugging Face transformers / vLLM / FlashAttention / FlashInfer / TensorRT-LLM 等基础设施，工程成本骤降。
3. **长上下文需求**：Prefix LM 的混合 mask 与 KV cache 优化（如 paged attention、prefix caching）耦合度低，移植成本高；切换到 Causal LM 后可享受现代推理框架的所有优化。
4. **对齐流水线**：DPO / RLHF / GRPO / RL-with-verifier 几乎清一色为 Causal LM 设计；继续坚持 Prefix LM 会在每个新 RL 算法上付出额外适配成本。
5. **推理范式（CoT / o1-like）**：思考链是天然的因果生成，Causal LM 与 Thinking Mode / Tool Use / Agent 的契合度最高。

### 6.3 GLM 路线"被取舍"的设计

- **2D Positional Encoding** 在 GLM-130B 即被 RoPE 替换；
- **Span shuffling** 在 ChatGLM2/3 被淡化；
- **Part A 双向 attention** 在 ChatGLM3 之后基本不再强调；
- **保留下来的 GLM 元素**：仅"GLM 品牌 + 训练数据/对齐 know-how + 工程框架（slime、SAT、SwissArmyTransformer）"。

可以理解为：**GLM 已从"特定预训练目标"演化为"团队/家族标识"**，现代 GLM-4.5 在架构上更接近 DeepSeek-MoE / Qwen3-MoE，而非最初的 GLM 论文。

---

## 七、关键创新点汇总

| # | 创新 | 出处 | 影响 |
|---|---|---|---|
| 1 | **自回归填空 (Autoregressive Blank Infilling)** | GLM 2021 / ACL 2022 | 用单一目标统一 NLU + 有/无条件生成；启发后续 UL2 等多任务预训练 |
| 2 | **2D 位置编码** | GLM 2021 | 解决 mask span 长度泄漏；后被 RoPE 取代 |
| 3 | **DeepNorm** | GLM-130B | 解决 100B+ 训练稳定性，被 BLOOM 等借鉴 |
| 4 | **INT4 量化无损推理（百亿级）** | GLM-130B | 首次让 130B 模型可在消费卡上推理（4×3090） |
| 5 | **Multi-Query Attention 国产化落地** | ChatGLM2-6B | 推动 MQA/GQA 在中文社区普及 |
| 6 | **All Tools Agent 范式** | GLM-4 | 模型自主决定调用 web/Python/T2I/自定义函数 |
| 7 | **Visual Expert** | CogVLM | 在每层 attention/FFN 并行视觉副本，深度融合而不损害纯文本能力 |
| 8 | **3D Causal VAE + Expert AdaLN** | CogVideoX | 长视频的高效 latent 表征 + 文本-视频深度融合 |
| 9 | **Relay Diffusion** | CogView3 | 级联扩散，低→高分辨率 relay |
| 10 | **Rumination（反刍）模型** | GLM-Z1-Rumination | 对标 Deep Research，工具调用 + RL 训练长程研究 |
| 11 | **Hybrid Reasoning (Thinking/Non-Thinking 单模型双模式)** | GLM-4.5 | 通过 API 参数即可切换；MoE 上较早实现 |
| 12 | **slime RL 框架** | GLM-4.5+ | 异步 rollout/trainer 解耦；GLM-4.5/4.6/4.7/5/5.1 共用 |
| 13 | **Loss-free Balance Routing**（在 GLM-4.5 MoE 中） | GLM-4.5 | 替代 auxiliary load loss，提升 MoE 训练稳定性 |
| 14 | **Preserved Thinking** | GLM-4.7 | 在多轮工具调用之间保留思考链，提升自主编程 |

---

## 八、对比小结

### 8.1 基座模型横向对比

| 模型 | 时间 | 架构类别 | 总参 / 激活 | 上下文 | 训练 tokens | 典型评测亮点 |
|---|---|---|---|---|---|---|
| GLM-130B | 2022.08 | Prefix LM (稠密) | 130B | 2K | 400B | LAMBADA SOTA、INT4 推理 |
| ChatGLM-6B | 2023.03 | Prefix LM (稠密) | 6.2B | 2K | ~1T | 中文对话开源里程碑 |
| ChatGLM2-6B | 2023.06 | Prefix LM + MQA | 6.2B | 32K | 1.4T | MMLU +23%、CEval +33% |
| ChatGLM3-6B | 2023.10 | ≈Causal LM + MQA | 6.2B | 32K / 128K | — | 工具/代码原生模板 |
| GLM-4-9B-Chat | 2024.06 | Causal LM | 9B | 128K / 1M | 10T | 接近 GPT-4 子项 |
| GLM-4-32B-0414 | 2025.04 | Causal LM (稠密) | 32B | 32K → 128K (YaRN) | — | BFCL 69.6、IFEval 87.6 |
| GLM-Z1-32B-0414 | 2025.04 | Causal LM + RL | 32B | 128K | — | 200 tok/s、AIME/MATH 接近 R1 |
| GLM-4.5 | 2025.07 | MoE + Hybrid Reasoning | 355B / 32B | 128K | 23T | 12 项基准均分全球第 2 |
| GLM-4.5-Air | 2025.07 | MoE | 106B / 12B | 128K | 23T | 超越 Gemini 2.5 Flash / Qwen3-235B |
| GLM-4.6 | 2025.09 | MoE | 357B / 32B | **200K** | — | 与 Claude Sonnet 4.5 近平价 |
| GLM-4.7 | 2025.12 | MoE + Preserved Thinking | 同代 | 200K | — | 自主编程方向 |

### 8.2 多模态模型横向对比

| 模型 | 时间 | 视觉编码器 | 融合方式 | LLM 底座 | 总参 | 亮点 |
|---|---|---|---|---|---|---|
| CogVLM-17B | 2023.11 | EVA-CLIP-E/14 | **Visual Expert（每层并行）** | Vicuna-7B | 17B | 10 项 SOTA |
| CogVLM2-19B | 2024.05 | SigLIP/EVA | Visual Expert | LLaMA3-8B | 19B | 1344² / 8K |
| GLM-4V-9B | 2024.06 | 同上 | Visual Expert | GLM-4-9B | ~13B | GLM 主家族绑定 |
| GLM-4.1V-9B-Thinking | 2025.07 | 视觉编码器 + MLP | **Encoder + Projector + LLM**（弃用 Visual Expert） | GLM-4-9B | 9B | RL Thinking 链 |
| GLM-4.5V | 2025.08 | 同上 | Encoder + Projector + LLM | GLM-4.5-Air (MoE) | 106B / 12B | 42 项基准 SOTA |
| GLM-4.6V / 4.6V-Flash | 2025 Q4–12 | 同上 | Encoder + Projector + LLM | GLM-4.6 / 9B | 357B / 9B | 200K + 端侧 |

---

## 九、参考文献

### 论文（按时间）

1. Ding M., Yang Z., Hong W., et al. **CogView: Mastering Text-to-Image Generation via Transformers**. NeurIPS 2021. arXiv:2105.13290. https://keg.cs.tsinghua.edu.cn/jietang/publications/NeurIPS21-Ding-et-al-CogView.pdf
2. Du Z., Qian Y., Liu X., Ding M., Qiu J., Yang Z., Tang J. **GLM: General Language Model Pretraining with Autoregressive Blank Infilling**. ACL 2022. arXiv:2103.10360. https://aclanthology.org/2022.acl-long.26/
3. Hong W., Ding M., Zheng W., Liu X., Tang J. **CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers**. ICLR 2023. arXiv:2205.15868.
4. Ding M., Zheng W., Hong W., Tang J. **CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformers**. NeurIPS 2022. arXiv:2204.14217.
5. Zeng A., Liu X., Du Z., et al. **GLM-130B: An Open Bilingual Pre-trained Model**. ICLR 2023. arXiv:2210.02414. https://arxiv.org/abs/2210.02414
6. Wang W., Lv Q., Yu W., et al. **CogVLM: Visual Expert for Pretrained Language Models**. NeurIPS 2024. arXiv:2311.03079. https://arxiv.org/abs/2311.03079
7. Zheng W., Teng J., Yang Z., et al. **CogView3: Finer and Faster Text-to-Image Generation via Relay Diffusion**. arXiv:2403.05121 (2024).
8. Team GLM (Zeng A., Xu B., et al.). **ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools**. arXiv:2406.12793 (2024). https://arxiv.org/abs/2406.12793
9. Hong W., Wang W., Ding M., et al. **CogVLM2: Visual Language Models for Image and Video Understanding**. arXiv:2408.16500 (2024). https://arxiv.org/abs/2408.16500
10. Yang Z., Teng J., Zheng W., et al. **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer**. ICLR 2025. arXiv:2408.06072. https://arxiv.org/abs/2408.06072
11. Team GLM-V. **GLM-4.1V-Thinking & GLM-4.5V & GLM-4.6V: Towards Versatile Multimodal Reasoning with a Family of VLMs**. arXiv:2507.01006 (2025).
12. Team GLM. **GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models**. arXiv:2508.06471 (2025-08).

### 官方资源

13. Z.AI 官网与开发者文档：https://z.ai ｜ https://docs.z.ai
14. 智谱 AI 开放平台 / bigmodel.cn：https://open.bigmodel.cn ｜ https://www.bigmodel.cn
15. Z.AI / 智谱 GitHub 组织：https://github.com/zai-org （含 GLM-130B、GLM-4、GLM-4.5、GLM-V、CogVLM2、CogVideo、CogView4、slime 等仓库）
16. THUDM GitHub 组织：https://github.com/THUDM （含 GLM、ChatGLM-6B、ChatGLM2-6B、ChatGLM3、GLM-4、slime 等历史仓库）
17. Hugging Face：https://huggingface.co/zai-org （GLM-4.6、CogVLM2、CogView 等权重）
18. GLM-130B 官方博客：https://keg.cs.tsinghua.edu.cn/glm-130b/
19. CogVLM2-Video 项目页：https://cogvlm2-video.github.io/
20. slime（RL 后训练框架）：https://github.com/THUDM/slime

### 第三方分析（参考）

21. Z.ai Blog. **GLM-4.5: Reasoning, Coding, and Agentic Abililties**. https://z.ai/blog/glm-4.5
22. APXML. *GLM-4.5 / GLM-4.6 Specifications and GPU VRAM Requirements*. https://apxml.com/models/glm-45 ｜ https://apxml.com/models/glm-46
23. The Decoder (2025-12). *Zhipu AI challenges Western rivals with low-cost GLM-4.7*.
24. vLLM 官方博客 (2025-08-19). *GLM-4.5 Meets vLLM: Built for Intelligent Agents*.

---

> 编纂说明：本报告以官方论文与发布信息为主要依据，对于社区流出的部分参数（如 GLM-4.7 / GLM-5 系列的精确激活/总参）已做标注；ChatGLM2/3 在"是否仍为 Prefix LM"上社区描述存在口径不一，本报告以官方仓库 README 与 ChatGLM 综述论文 (arXiv:2406.12793) 的最终归纳为准，并在第六章给出渐进式转型的判断。
