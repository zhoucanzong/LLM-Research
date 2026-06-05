# Doubao/Seed系列模型深度调研报告

## 概述与时间线

字节跳动（ByteDance）旗下的AI大模型研发主要由**Seed团队**（2023年正式组建）主导，产品侧通过**豆包（Doubao）**品牌面向消费者和企业用户。Seed团队定位为"探索通用智能的无限可能"，其研究覆盖语言模型、多模态理解与生成、语音、图像、视频等全方向。

### 关键时间线

| 时间 | 事件 |
|------|------|
| 2023年 | ByteDance Seed团队正式组建，豆包App上线 |
| 2024年6月 | Seed-TTS论文发布（语音合成基础模型） |
| 2024年7月 | Seed-ASR论文发布（LLM-based语音识别） |
| 2024年9月 | PixelDance与Seaweed视频生成模型发布 |
| 2024年11月 | PixelDance/Seaweed在即梦AI上线 |
| 2025年1月 | Doubao-1.5-pro发布（MoE架构基座模型） |
| 2025年4月 | Seedream 3.0图像生成模型发布 |
| 2025年4月 | Seed-Thinking-v1.5推理模型技术报告发布 |
| 2025年5月 | Seed1.5-VL视觉语言模型发布 |
| 2025年5月 | BAGEL统一多模态模型开源 |
| 2025年6月 | Seed-Coder代码模型开源 |
| 2025年6月 | Seedance 1.0视频生成模型技术报告发布 |
| 2025年下半年 | Seedream 4.0/4.5图像生成模型发布 |
| 2025年12月 | Seedance 1.5 Pro（音画同步视频生成）上线即梦 |
| 2026年2月 | Seed 2.0系列（Pro/Lite/Mini）正式发布 |
| 2026年2月 | Seedance 2.0视频生成模型发布 |
| 2026年3月 | Seed1.8 Agent模型发布 |

---

## 一、基座语言模型演进

### 1.1 早期阶段：Doubao基座模型

字节跳动在2023年推出豆包App，底层大模型即为内部研发的Seed系列基座。早期版本信息公开有限，已知的是：

- 模型通过字节内部**50+业务场景**实践验证
- 每日处理**千亿级tokens**使用量
- 通过火山引擎（Volcano Engine）对外提供API服务

> **注**：早期Seed-LLM的具体架构（Dense vs MoE、参数规模）未有官方详细披露。

### 1.2 Doubao-1.5-pro（2025年1月）

Doubao-1.5-pro是字节跳动第一次大规模公开披露技术细节的基座模型，于2025年1月22日发布。

**架构特征：**
- **MoE（Mixture-of-Experts）架构**：采用高度稀疏的MoE设计
- **训练-推理一体化设计**：从预训练阶段即考虑推理效率
- **性能杠杆达到7倍**：激活参数仅为等效Dense模型参数量的1/7，即可超过Dense模型性能（此前业界普遍水平不到3倍）
- 训练数据量超过9T tokens（中间版本），最终版本更大
- 性能超过Llama3.1-405B等超大稠密模型

**推理系统优化：**
- Prefill/Decode分离的Serving系统
- 四象限异构硬件策略（Prefill Attention/FFN、Decode Attention/FFN）
- Prefill阶段Tensor Core利用率接近60%
- FlashAttention 8-bit实现 + Per N tokens Per Sequence量化
- Decode FFN采用W4A8量化 + EP部署
- 支持Prefill和Decode集群灵活配比和动态扩缩

**Post-Training特色：**
- 完全自主数据生产体系，**不使用任何其他模型数据**
- SFT阶段：算法驱动的训练数据优化 + 模型自演进（Self-evolve）
- Reward Model：融合合成与挖掘数据，多阶段训练框架
- RL阶段：基于veRL框架，高并行化多角色训练推理一体框架
- 提出生成式RM建模方法（区别于传统判别式RM）

**多模态能力（集成于同一模型）：**
- **视觉多模态**：原生动态分辨率架构，自研Doubao ViT（2.4B参数，综合SOTA）
- **语音多模态**：Speech2Speech端到端框架，语音与文本Token融合

**推理能力（深度思考模式）：**
- Doubao-1.5-pro-AS1-Preview在AIME上超过O1-preview和O1
- 完全不使用其他模型数据，通过RL Scaling实现

### 1.3 Seed 2.0系列（2026年2月14日）

Seed 2.0是字节跳动新一代旗舰模型家族，正式发布于2026年2月14日，定位为"面向真实世界应用的智能前沿"。

**模型家族：**

| 模型 | 定位 | 核心优势 |
|------|------|----------|
| Seed 2.0 Pro | 旗舰模型 | 极致性能与智能天花板 |
| Seed 2.0 Lite | 效率标杆 | 性能、速度与成本的平衡 |
| Seed 2.0 Mini | 轻量先锋 | 高并发低延迟 |

**核心性能指标（Seed 2.0 Pro）：**
- AIME 2025: **98.3**（与GPT-5.2、Gemini 3 Pro直接竞争）
- HMMT Feb: **97.3**
- GPQA Diamond: **88.9**
- Codeforces: **3020**（国际竞赛金牌水平）
- SWE-Bench Verified: **76.5**
- VideoMME: **89.5**
- LMSYS Chatbot Arena: Text Arena第6名，Vision Arena第3-4名

**架构推断：**
- 延续MoE架构路线（从Seed-Thinking-v1.5的20B active/200B total参数设计可推断）
- 原生多模态：支持文本、图像、视频输入
- 全面支持Agent能力（工具调用、网页浏览、终端操作）

> **注**：Seed 2.0完整技术细节（具体参数规模、专家数量等）未全部公开披露，以上部分为根据已知信息的合理推断。

### 1.4 UltraMem：超稀疏架构探索

字节Seed团队在模型架构研究方面的一个重要创新是**UltraMem（Ultra-Sparse Memory Network）**，发表于ICLR 2025：

- 提出超稀疏内存层，有效解决MoE推理中的高推理成本和内存访问问题
- 实现2-6倍推理加速
- 相比MoE降低高达83%的推理成本
- 这一研究为Seed系列模型的高效推理提供了技术储备

---

## 二、推理模型：Seed-Thinking

### 2.1 Seed-Thinking-v1.5（2025年4月）

Seed-Thinking-v1.5是字节跳动Seed团队推出的推理模型，2025年4月14日公开技术报告，4月17日通过火山引擎开放API。

**模型架构：**
- **MoE架构**
- 总参数：**200B**
- 激活参数：**20B**
- 相比DeepSeek R1推理成本降低50%

**核心性能：**
- AIME 2024: **86.7**（追平OpenAI o3-mini-high）
- Codeforces pass@8: **55.0%**（接近Gemini 2.5 Pro）
- GPQA: **77.3%**（接近o3-mini-high）
- 人类评估超过DeepSeek R1约8%

**训练方法论——四大技术支柱：**

#### （1）数据系统：融合可验证与创造性数据
- **可验证数据**（数学/代码）：三重清洗（人工审核→模型过滤→多模型验证），从百万样本中保留10万高难度问题
- **不可验证数据**（创意写作）：基于Doubao 1.5 Pro训练集，双向奖励方法优化
- **新基准BeyondAIME**：100道未解答STEM题目，解决现有测试区分度不足问题

#### （2）奖励模型：双轨系统
- **可验证任务**：两代验证器（Seed-Verifier → Seed-Thinking-Verifier），从字符匹配升级到逐行对比推理步骤，准确率超99%
- **不可验证任务**：成对对比训练，数千万"AB测试"捕捉人类隐含偏好
- **双轨融合**：硬指标（对错）与软偏好（优劣）互补

#### （3）训练方法：SFT + RL两阶段
- **SFT阶段**：40万高质量实例（30万可验证 + 10万不可验证），构建长思维链数据集
- **RL阶段**：
  - 三重数据引擎（可验证/通用/混合数据）
  - 算法创新：价值预训练、解耦GE等
  - 在线数据适应技术
  - 解决训练不稳定和长链推理断层问题

#### （4）训练框架
- **HybridFlow编程模型**：支持算法快速探索与分布式并行
- **流式推理系统（SRS）**：解耦模型演进与异步推理，训练速度提升3倍
- **三层并行架构**：Tensor/Expert/Serial并行 + KARP算法优化GPU利用率
- 万亿参数下稳定性达95%

### 2.2 与同期模型对比

| 维度 | Seed-Thinking-v1.5 | DeepSeek-R1 | OpenAI o3-mini |
|------|-------|-------|--------|
| 架构 | MoE 200B/20B active | MoE 671B/37B active | 未公开 |
| AIME 2024 | 86.7 | ~87 | 86.7(high) |
| 推理成本 | 基准 | 约2倍 | 未公开 |
| 训练方法 | SFT+RL双阶段 | 纯RL + 蒸馏 | 未公开 |
| 特色 | 双轨奖励、BeyondAIME | 纯RL涌现 | 推理scaling |

**关键差异化：**
- Seed-Thinking采用"SFT奠基 + RL磨练"路线，而非DeepSeek-R1的纯RL路线
- 更紧凑的模型规模（20B active vs 37B active）带来显著成本优势
- 双轨奖励系统同时覆盖可验证和不可验证任务

---

## 三、代码模型：Seed-Coder

### 3.1 概述（2025年6月）

Seed-Coder是字节Seed团队开源的代码大语言模型系列，核心理念是**"让代码模型为自己策划数据"（Let the Code Model Curate Data for Itself）**。

**模型规模：** 8B参数
**模型变体：** Base / Instruct / Reasoning三个版本

### 3.2 核心创新：模型中心化数据管线

Seed-Coder最大的创新在于其数据处理方法——**最小化人工参与的模型中心化（Model-Centric）数据管线**：

- 主要使用LLM（而非手工规则）进行代码数据评分和过滤
- LLM能捕捉难以量化的代码质量细微标准
- 可扩展地、一致性地处理数十亿样本

**数据规模：** 6万亿（6T）tokens的代码预训练语料

**数据来源四类：**
1. **文件级代码**：来自GitHub的单个代码文件
2. **仓库级代码**：保留项目结构的代码文件
3. **Commits数据**：GitHub Commits快照，包含消息、元数据、文件和补丁
4. **代码相关网页数据**：Web归档中包含代码块或高度代码相关的文档

### 3.3 LLM质量过滤器

提出文件级评分模型，一次性过滤低质量代码文件，评估四个维度：
- **可读性（Readability）**：注释合理、命名一致、格式规范
- **模块性（Modularity）**：结构良好、避免过于复杂的函数
- **清晰度（Clarity）**：最小化冗余、清晰传达意图
- **可复用性（Reusability）**：无语法/逻辑错误、易于集成

### 3.4 训练策略

- **两阶段预训练**：
  - 常规预训练：文件级代码 + 代码相关网页数据
  - 持续预训练：全四类数据，增强长上下文能力
- **并行化管线设计**：解耦各过滤模块的顺序依赖，支持增量数据扩展
- **Fill-in-the-Middle（FIM）格式**应用于两个训练阶段

### 3.5 性能表现

Seed-Coder-8B在同规模模型中达到SOTA水平，覆盖：
- 代码生成
- 代码解释
- 代码调试
- 代码编辑
- 真实世界软件工程任务

---

## 四、语音模型

### 4.1 Seed-TTS：语音合成基础模型（2024年6月）

Seed-TTS是字节跳动发布的大规模自回归文本到语音（TTS）模型家族，论文发表于2024年6月（arXiv: 2406.02430）。

**系统架构（推理管线四阶段）：**
1. **Speech Tokenizer**：从参考语音学习tokens
2. **自回归语言模型**：基于条件文本和语音生成speech tokens
3. **Diffusion Transformer**：以粗到精方式从speech tokens生成连续语音表示
4. **声学Vocoder**：从扩散输出产生高质量语音

**技术路线：自回归 + 扩散模型的混合架构**

**核心能力：**
- **零样本语音克隆（Zero-shot In-context Learning）**：仅需少量参考语音即可克隆说话人特征
- **跨语言生成**：支持中英文互换生成，保持说话人音色
- **说话人微调（Speaker Fine-tune）**：少量数据即可生成高质量定制语音
- **情感控制**：支持愤怒、快乐、悲伤、温柔、困惑、恐惧等情感
- **语音分解（Speech Factorization）**：零样本语音转换
- **强化学习偏好对齐**：通过RL进行情感控制优化
- **全扩散语音生成**：提供纯Diffusion-based方案作为替代
- **内容编辑**：修改语音中的特定词汇而保持整体音色
- **语速编辑**：精确控制生成语音的速度

**应用场景：**
- 有声书（多说话人）
- 跨语言内容创作（带口型编辑）
- 豆包AI对话语音

**成就：** 在说话人相似度和自然度方面达到接近人类水平的性能。

### 4.2 Seed-ASR：语音识别（2024年7月）

Seed-ASR是基于大语言模型的语音识别模型，论文发表于2024年7月（arXiv: 2407.04675）。

**架构特点：**
- 基于"音频条件化LLM"框架：将连续语音表示输入LLM，结合文本指令进行转录
- 支持上下文感知（Context-Aware）识别

**上下文感知能力：**
- 对话历史上下文：利用前文对话纠正同音词
- Agent名称上下文：识别特定角色名称
- Agent描述信息：利用角色描述提升识别准确度
- 修改历史记录：学习用户纠正模式，避免重复错误
- 会议参与者名称：利用参会者列表提升人名识别

**多样性支持：**
- 多领域（直播、美食、游戏等）
- 多方言（吴语、粤语、四川话、湖南话等）
- 多口音
- 多语言（中、英、印尼语、葡萄牙语等）
- 强噪声环境

**已部署到豆包App的语音对话功能。**

### 4.3 Seeduplex：全双工语音模型

Seeduplex是Seed团队推出的原生全双工语音LLM（Native Full-Duplex Speech LLM），相比上一代豆包端到端语音模型，实现了真正的"边听边说"能力，通过预训练和对齐技术，实现低延迟语音到语音生成。

---

## 五、图像生成模型：Seedream系列

### 5.1 Seedream 3.0（2025年4月）

Seedream 3.0是字节跳动推出的高性能中英双语图像生成基础模型。

**核心特性：**
- **原生高分辨率**：支持原生高分辨率生成
- **中英双语原生支持**：深度理解中文语义
- **3秒快速生成**：面向海报设计和创意视觉，可在约3秒内生成高质量图像
- 基于Diffusion模型架构

### 5.2 Seedream 4.0（2025年下半年）

Seedream 4.0是新一代多模态图像创作模型，实现了**生成与编辑能力的统一架构整合**。

**核心创新：**
- **统一架构**：将Text-to-Image（T2I）生成和图像编辑集成在单一统一架构中
- **高分辨率支持**：最高支持4K分辨率
- **多模态输入**：支持文本、图像等多种输入条件

**技术路线：** 基于Diffusion Transformer架构

**性能：** 声称在图像生成和编辑方面超过Gemini 2.5 Flash Image（基于内部评估基准）。

### 5.3 Seedream 4.5

Seedream 4.5在4.0基础上进一步升级：
- 精确识别多张输入图像中的目标元素
- 支持可控且一致的多图像生成
- 增强的一致性保持能力

---

## 六、视频生成模型

### 6.1 PixelDance与Seaweed（2024年9月-11月）

2024年9月，字节跳动通过火山引擎发布了两款视频生成模型：

**PixelDance：**
- 增强多主体交互能力
- 流畅的角色动作
- 2024年11月在即梦AI上线

**Seaweed：**
- 多摄像机视频生成
- 与PixelDance互补的技术路线
- 采用3D Multi-modal RoPE位置编码

> 这两款模型标志着字节跳动正式进入AI视频生成竞赛。

### 6.2 Seedance 1.0（2025年6月）

Seedance 1.0是字节Seed团队的视频生成基础模型，2025年6月发布技术报告并集成到豆包和即梦平台。

**架构设计：**
- **VAE**：时序因果卷积架构（Temporally-Causal Compression）
  - 压缩比：时间4x，空间16x16
  - 潜在通道数C=48
  - 训练使用L1重建损失 + KL损失 + LPIPS感知损失 + 对抗训练损失
- **Diffusion Transformer（DiT）**：
  - 解耦空间层与时间层（Decoupled Spatial and Temporal Layers）
  - 空间层：帧内注意力 + 文本交叉注意力
  - 时间层：跨帧注意力 + 窗口分割
  - MMDiT架构（类似Stable Diffusion 3）
  - 文本编码器：基于fine-tuned decoder-only LLM
- **多镜头MM-RoPE**：支持交错的视觉/文本token序列，原生多镜头叙事
- **统一任务公式化**：通过通道拼接和二值掩码统一T2I/T2V/I2V任务
- **级联扩散框架**：Base模型生成480p → Diffusion Refiner上采样至720p/1080p

**Prompt Engineering模块：**
- 基于Qwen2.5-14B初始化
- SFT + DPO两阶段训练
- 将用户提示转换为密集视频描述

**后训练优化：**
- 精选少量高质量数据进行SFT
- 视频定制RLHF算法
- 多个奖励模型驱动的反馈学习

**推理加速：**
- 多阶段蒸馏框架
- 减少NFE（函数评估次数）
- 实现10倍以上端到端加速
- 5秒1080p视频仅需41.4秒（NVIDIA L20）

**核心特色：**
- 中英双语原生支持
- 综合生成能力：运动流畅性、物理合理性、真实感
- 精确指令遵循：多主体交互、自适应相机控制
- 原生多镜头叙事能力

### 6.3 Seedance 1.5 Pro（2025年12月）

Seedance 1.5 Pro在即梦网页版上线，核心升级为**音画同步视频生成**：
- 独家音频-视频联合生成技术
- 人物口型同步
- 乐器演奏同步
- 自然音效对齐

### 6.4 Seedance 2.0（2026年2月）

Seedance 2.0是字节最新的视频生成旗舰模型。

**架构升级：**
- **统一多模态音视频联合生成架构**
- 支持文本、图像、音频、视频多种输入
- 行业最全面的多模态内容参考和编辑能力

**核心特性：**
- 卓越的运动稳定性
- 音视频联合生成
- 超真实沉浸式体验
- 在SeedVideoBench-2.0多维度评估中领先

### 6.5 即梦AI（产品层）

即梦AI是字节跳动面向消费者的AI创作工具平台，集成了上述视频生成模型：
- 支持文字生成视频
- 支持图片生成视频
- 免费对公众开放使用
- 2025年底-2026年全面升级，底层模型从PixelDance/Seaweed升级为Seedance系列

---

## 七、多模态理解模型

### 7.1 Seed1.5-VL（2025年5月）

Seed1.5-VL是字节Seed团队的视觉语言基础模型，设计目标是推进通用多模态理解与推理。

**架构：**
- **视觉编码器**：532M参数
- **LLM主干**：MoE架构，**20B激活参数**
- 相对紧凑的架构实现顶级性能

**能力覆盖：**
- 通用多模态理解与推理
- GUI控制/Agent能力
- 视频理解
- 多种视觉推理任务SOTA

**开源发布：** GitHub（ByteDance-Seed/Seed1.5-VL）

### 7.2 BAGEL：统一多模态基础模型（2025年5月）

BAGEL是字节Seed团队开源的统一多模态基础模型，支持理解与生成的统一。

**模型规格：**
- 7B激活参数（14B总参数）
- 基于大规模交错多模态数据训练
- 预训练从LLM初始化

**核心能力：**
- **统一理解与生成**：单一模型同时支持多模态理解和图像生成
- 支持文本、图像的统一处理
- 图像生成与编辑
- 图像理解与推理

**架构创新：** 论文标题"Emerging Properties in Unified Multimodal Pretraining"揭示了统一多模态预训练中涌现的属性。

**开源地址：** GitHub（ByteDance-Seed/BAGEL）

### 7.3 Seed1.8：通用Agent模型（2026年3月）

Seed1.8定位为"面向通用现实世界Agent能力"的基础模型。

**设计理念：**
- 超越单轮预测，支持多轮交互和任务执行
- 保留核心LLM和VLM能力的同时扩展Agent能力
- 工作流导向的MaaS（Model-as-a-Service）基础

**能力特征：**
- 支持文本和图像输入
- 强大的工具调用能力
- 多步骤高精度任务执行
- 标准LLM/VLM基准保持竞争力
- 推理、复杂指令遵循、知识任务表现优异

**应用方向：**
- 网页浏览自动化
- 终端操作
- 真实世界多轮任务

**开源发布：** GitHub（ByteDance-Seed/Seed-1.8）

### 7.4 Seed-X：多语言翻译模型（2025年）

值得注意的是，"Seed-X"这一名称在字节体系中指代的是多语言翻译LLM系列（而非早期传闻的通用多模态模型）：
- 7B参数规模
- 包含Instruct和Reasoning模型
- 推进翻译能力的极限
- 开源发布（含奖励模型Seed-X-RM-7B）

---

## 八、跨方向技术演进分析

### 8.1 架构统一性：MoE贯穿全线

字节Seed系列在语言模型方向坚定采用**MoE架构**：
- Doubao-1.5-pro：高稀疏MoE，性能杠杆7倍
- Seed-Thinking-v1.5：200B total / 20B active
- Seed1.5-VL：20B active MoE LLM
- Seed 2.0系列：延续MoE路线（推断）

**核心优势：** 以较小的激活参数实现接近超大Dense模型的性能，同时保持推理成本可控。

### 8.2 训练方法论：RL全面渗透

强化学习在Seed系列各方向中无处不在：
- **语言模型RL**：Doubao-1.5-pro的后训练、Seed-Thinking的推理RL
- **语音RL**：Seed-TTS通过RL进行偏好对齐和情感控制
- **视频RL**：Seedance 1.0的视频RLHF
- **Prompt Engineering RL**：Seedance中PE模块的DPO训练

### 8.3 数据自主性：不依赖外部模型数据

字节Seed团队多次强调**数据自主可控**：
- Doubao-1.5-pro："坚持不走捷径，不使用任何其他模型的数据"
- Seed-Thinking-v1.5："完全不使用其他模型数据"
- Seed-Coder：模型自策划数据管线

这一策略虽然增加了初始成本，但确保了数据独立性和避免知识蒸馏的潜在法律风险。

### 8.4 从单模态到统一多模态

演进路径清晰可见：
1. **单独的语言/视觉/语音模型** → 
2. **Doubao-1.5-pro集成多模态** → 
3. **BAGEL统一理解与生成** → 
4. **Seed 2.0原生多模态** → 
5. **Seed1.8 Agent能力整合**

### 8.5 推理效率的极致追求

从架构层到系统层的全栈优化：
- **架构层**：UltraMem超稀疏、MoE高稀疏比
- **算法层**：蒸馏、量化（W4A8）
- **系统层**：PD分离、异构硬件、定制网卡和协议
- **应用层**：Speculative Decoding、动态扩缩容

---

## 九、关键创新点总结

1. **7倍MoE性能杠杆**：通过模型结构和训练算法优化，将MoE性能杠杆从业界3倍提升至7倍

2. **双轨奖励系统**：同时处理可验证（对错判断）和不可验证（偏好优劣）任务的统一奖励框架

3. **模型中心化代码数据管线**：Seed-Coder让模型自己策划训练数据，最小化人工参与

4. **自回归+扩散混合语音架构**：Seed-TTS四阶段管线实现接近人类水平的零样本语音克隆

5. **解耦时空DiT架构**：Seedance的空间/时间层解耦 + 窗口注意力，兼顾质量与效率

6. **统一多模态预训练**：BAGEL揭示了统一预训练中的涌现属性

7. **全栈推理优化**：从UltraMem到PD分离Serving的系统性推理加速方案

8. **数据完全自主可控**：从预训练到后训练全程不依赖其他模型数据

9. **音视频联合生成**：Seedance 2.0实现统一的音频-视频联合生成架构

10. **Agent能力泛化**：Seed1.8从语言理解扩展到通用现实世界Agent

---

## 十、整体生态布局

字节跳动通过Seed团队在AI各方向构建了完整生态：

```
                    ┌─────────────────────────────────┐
                    │     Seed 2.0 (旗舰基座)          │
                    │  Pro / Lite / Mini               │
                    └──────────┬──────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼─────┐         ┌─────▼────┐          ┌─────▼─────┐
   │ 推理增强  │         │ 多模态    │          │ 生成模型   │
   ├──────────┤         ├──────────┤          ├───────────┤
   │Seed-     │         │Seed1.5-VL│          │Seedream   │
   │Thinking  │         │BAGEL     │          │(图像)     │
   │v1.5      │         │Seed1.8   │          │Seedance   │
   │          │         │          │          │(视频)     │
   └──────────┘         └──────────┘          │Seed-TTS   │
                                              │(语音)     │
        ┌──────────────┐                      └───────────┘
        │ 代码专项      │
        ├──────────────┤              ┌─────────────────┐
        │Seed-Coder    │              │  产品层          │
        └──────────────┘              ├─────────────────┤
                                      │ 豆包App          │
                                      │ 即梦AI           │
                                      │ 火山引擎API      │
                                      └─────────────────┘
```

**Doubao产品与模型的关系：**
- **豆包App**：面向C端用户的AI助手，底层集成Seed系列全线模型能力
- **即梦AI**：面向创作者的AI生成工具，集成Seedance/Seedream等生成模型
- **火山引擎**：面向B端的模型服务平台，提供全系列模型API
- 模型命名规则：API层面使用"Doubao-"前缀（如Doubao-Seed-2.0-pro），研究层面使用"Seed"品牌

---

## 参考文献

1. ByteDance Seed Team. "Seed-TTS: A Family of High-Quality Versatile Speech Generation Models." arXiv:2406.02430, 2024.

2. ByteDance Seed Team. "Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition." arXiv:2407.04675, 2024.

3. ByteDance Seed Team. "Seed-Thinking-v1.5: Advancing Superb Reasoning Models with Reinforcement Learning." arXiv:2504.13914, 2025.

4. ByteDance Seed Team. "Seed-Coder: Let the Code Model Curate Data for Itself." arXiv:2506.03524, 2025.

5. ByteDance Seed Team. "Seed1.5-VL Technical Report." arXiv:2505.07062, 2025.

6. ByteDance Seed Team. "Emerging Properties in Unified Multimodal Pretraining (BAGEL)." arXiv:2505.14683, 2025.

7. ByteDance Seed Team. "Seedance 1.0: Exploring the Boundaries of Video Generation Models." arXiv:2506.09113, 2025.

8. ByteDance Seed Team. "Seedream 3.0 Technical Report." arXiv:2504.11346, 2025.

9. ByteDance Seed Team. "Seedream 4.0: Toward Next-generation Multimodal Image Generation." arXiv:2509.20427, 2025.

10. ByteDance Seed Team. "Seed1.8 Model Card: Towards Generalized Real-World Agency." arXiv:2603.20633, 2026.

11. ByteDance Seed Team. "Seed2.0 Model Card: Towards Intelligence Frontier for Real-World Applications." Technical Report, 2026.

12. ByteDance Seed Team. "Ultra-Sparse Memory Network (UltraMem)." ICLR 2025. arXiv:2411.12364.

13. ByteDance Seed Team. "Doubao-1.5-pro Technical Blog." seed.bytedance.com, January 2025.

14. ByteDance Seed Team. "Seedance 2.0." seed.bytedance.com, 2026.

15. ByteDance Seed Team. "Seeduplex: Native Full-Duplex Speech LLM." research.doubao.com, 2025.

---

*报告撰写日期：2025年6月*
*信息来源：公开论文、官方技术博客、GitHub仓库、新闻报道*
*注：部分模型的具体参数规模和内部架构细节未完全公开，报告中已标注哪些为确认信息、哪些为基于公开信息的合理推断。*
