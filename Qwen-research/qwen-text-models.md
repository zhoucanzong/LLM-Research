# Qwen纯文本模型系列发展脉络

## 概述

Qwen（通义千问）是阿里巴巴云推出的大语言模型系列，自2023年8月首次发布以来，经历了多代重大迭代。本报告梳理Qwen纯文本语言模型从初代到Qwen3.7的完整技术演进，涵盖模型架构、训练策略、规模系列和关键创新。

> 最后更新：2026年6月（增加Qwen3.5/3.6/3.7/Next/Coder系列信息）

---

## 1. Qwen（初代，2023年8月—12月）

### 1.1 发布背景

Qwen初代于2023年8月开始陆续开源，技术报告于2023年9月28日在arXiv发布（arXiv:2309.16609）。模型系列包括基座模型Qwen和对话模型Qwen-Chat，覆盖多个参数规模。

### 1.2 模型架构

Qwen初代采用标准的Decoder-only Transformer架构，核心设计选择如下：

| 组件 | 技术方案 | 说明 |
|------|----------|------|
| 注意力机制 | Multi-Head Attention (MHA) | 每个Query头拥有独立的KV头 |
| 位置编码 | RoPE (Rotary Position Embedding) | 旋转位置嵌入 |
| FFN激活函数 | SwiGLU | 门控线性单元变体 |
| 归一化 | RMSNorm + Pre-Normalization | 前置归一化提升训练稳定性 |
| 注意力偏置 | QKV Bias | Q/K/V投影均含bias项 |
| 词表大小 | 151,936 | 基于Byte-level BPE (BBPE) |
| Embedding绑定 | 不绑定 (Untied) | 输入输出embedding分离 |

### 1.3 模型规模系列

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头数 | 中间层维度 | 上下文长度 |
|------|--------|------|----------|------------|------------|-----------|
| Qwen-1.8B | 1.8B | 24 | 2,048 | 16 | 5,504 | 8K |
| Qwen-7B | 7B | 32 | 4,096 | 32 | 11,008 | 8K |
| Qwen-14B | 14B | 40 | 5,120 | 40 | 13,696 | 8K |
| Qwen-72B | 72B | 80 | 8,192 | 64 | 24,576 | 32K |

### 1.4 训练策略

- **预训练数据**：约3万亿tokens，覆盖公开网页文本、书籍、代码、百科等
- **分词器**：BBPE分词，词表大小151,936，针对中英文优化，编码效率高
- **上下文扩展**：基础训练上下文2048/8192，通过NTK-aware interpolation + LogN-Scaling + Window Attention扩展至32K
- **对齐方法**：SFT (Supervised Fine-Tuning) + RLHF (基于PPO的强化学习)

### 1.5 关键创新

- 首个中英双语大模型系列，在中文理解任务上表现优异
- BBPE大词表设计（151,936），相比LLaMA的32K词表，对中文和多语言编码效率显著提升
- QKV bias的引入为后续Qwen架构奠定基础
- 强大的工具调用和代码解释器能力

---

## 2. Qwen1.5（2024年2月）

### 2.1 发布背景

Qwen1.5于2024年2月发布，定位为Qwen系列的1.5版本迭代。未发布单独的技术论文，以官方博客形式公布。核心目标是提升对齐质量和开发者体验。

### 2.2 模型架构

Qwen1.5保持与Qwen初代相同的基础架构，主要架构特征不变：

| 组件 | 技术方案 | 与初代差异 |
|------|----------|-----------|
| 注意力机制 | MHA（小模型）/ GQA（110B） | 110B模型引入GQA |
| 位置编码 | RoPE | 不变 |
| FFN激活函数 | SwiGLU | 不变 |
| 归一化 | RMSNorm + Pre-Normalization | 不变 |
| 注意力偏置 | QKV Bias | 不变 |
| 词表大小 | 151,936 (BBPE) | 不变 |
| 上下文长度 | 32,768 (所有模型) | 统一扩展至32K |

### 2.3 模型规模系列

| 模型 | 参数量 | 注意力类型 | 上下文长度 |
|------|--------|-----------|-----------|
| Qwen1.5-0.5B | 0.5B | MHA | 32K |
| Qwen1.5-1.8B | 1.8B | MHA | 32K |
| Qwen1.5-4B | 4B | MHA | 32K |
| Qwen1.5-7B | 7B | MHA | 32K |
| Qwen1.5-14B | 14B | MHA | 32K |
| Qwen1.5-32B | 32B | MHA | 32K |
| Qwen1.5-72B | 72B | MHA | 32K |
| Qwen1.5-110B | 110B | GQA | 32K |
| Qwen1.5-MoE-A2.7B | 14.3B (2.7B激活) | MHA | 32K |

### 2.4 训练策略

- **预训练数据**：约3万亿tokens（与Qwen初代规模相近，但进行了数据质量优化）
- **对齐方法**：SFT + DPO (Direct Preference Optimization) + PPO
- **关键改进**：对chat模型的人类偏好对齐进行了大幅改善

### 2.5 关键创新与变更

- **统一32K上下文**：所有模型均支持32K上下文长度
- **多语言增强**：支持更多语言的理解和生成
- **首个MoE模型**：Qwen1.5-MoE-A2.7B，总参数14.3B，每token仅激活2.7B，引入细粒度专家和共享专家机制
- **开发者体验改善**：代码合并至Hugging Face transformers (≥4.37.0)，无需trust_remote_code
- **生态兼容**：vLLM、SGLang、llama.cpp、Ollama等框架全面支持
- **110B模型**：首次在大尺寸模型中引入GQA

---

## 3. Qwen2（2024年6月）

### 3.1 发布背景

Qwen2于2024年6月发布，技术报告同月在arXiv发布（arXiv:2407.10671）。这是Qwen系列的一次重大架构升级，引入了多项关键改动。

### 3.2 模型架构

| 组件 | 技术方案 | 与Qwen1.5差异 |
|------|----------|--------------|
| 注意力机制 | **GQA (所有模型)** | 从MHA全面升级为GQA |
| 位置编码 | RoPE (base freq 1,000,000) | 基频从10,000提升至1,000,000 |
| 长上下文 | DCA (Dual Chunk Attention) + YARN | 新增，支持128K推理 |
| FFN激活函数 | SwiGLU | 不变 |
| 归一化 | RMSNorm + Pre-Normalization | 不变 |
| 注意力偏置 | QKV Bias | 保留 |
| 词表大小 | 151,646 (BBPE) | 微调（从151,936变为151,646） |
| Embedding绑定 | 小模型绑定，大模型不绑定 | 新策略 |

### 3.3 模型规模系列（详细架构参数）

| 配置 | Qwen2-0.5B | Qwen2-1.5B | Qwen2-7B | Qwen2-72B | Qwen2-57B-A14B (MoE) |
|------|-----------|-----------|---------|---------|---------------------|
| 隐藏维度 | 896 | 1,536 | 3,584 | 8,192 | 3,584 |
| 层数 | 24 | 28 | 28 | 80 | 28 |
| Q头数 | 14 | 12 | 28 | 64 | 28 |
| KV头数 | 2 | 2 | 4 | 8 | 4 |
| Head Size | 64 | 128 | 128 | 128 | 128 |
| 中间层维度 | 4,864 | 8,960 | 18,944 | 29,568 | 2,560 (每专家) |
| 路由专家数 | - | - | - | - | 64 |
| 激活专家数 | - | - | - | - | 8 |
| 共享专家数 | - | - | - | - | 8 |
| Embedding绑定 | Yes | Yes | No | No | No |
| 训练tokens | 12T | 7T | 7T | 7T | 4.5T |
| 上下文长度 | 32K→128K | 32K→128K | 32K→128K | 32K→128K | 32K→128K |

### 3.4 训练策略

**预训练**：
- 数据规模从Qwen1.5的3T扩展至7T tokens（0.5B模型使用12T）
- 数据质量增强：模型辅助过滤低质量数据，合成高质量预训练数据
- 大幅扩充代码、数学和多语言数据，支持约30种语言
- 训练后期上下文从4K扩展至32K

**后训练**：
- SFT：50万+样本，覆盖指令遵循、代码、数学、推理、角色扮演、多语言、安全
- RLHF分两阶段：
  - 离线阶段：DPO (Direct Preference Optimization)
  - 在线阶段：基于奖励模型的Online DPO + Online Merging Optimizer

### 3.5 关键创新

- **GQA全面应用**：所有模型规模均采用GQA，大幅降低KV cache内存占用
- **DCA + YARN**：支持最长128K tokens的推理上下文
- **MoE架构**：Qwen2-57B-A14B采用细粒度专家（fine-grained experts）+ 共享专家设计
- **数据规模飞跃**：从3T到7T/12T tokens
- **多语言支持**：从中英为主扩展到约30种语言

---

## 4. Qwen2.5（2024年9月）

### 4.1 发布背景

Qwen2.5于2024年9月发布，技术报告于2024年12月在arXiv发布（arXiv:2412.15115）。这是一次全面升级，在预训练和后训练阶段均有显著改进。

### 4.2 模型架构

Qwen2.5在架构基础上与Qwen2保持一致，核心设计不变：

| 组件 | 技术方案 | 与Qwen2差异 |
|------|----------|------------|
| 注意力机制 | GQA | 不变 |
| 位置编码 | RoPE | 不变 |
| 长上下文 | DCA + YARN | 更多模型支持128K |
| FFN激活函数 | SwiGLU | 不变 |
| 归一化 | RMSNorm + Pre-Normalization | 不变 |
| 注意力偏置 | QKV Bias | 保留 |
| 词表大小 | 151,646 (BBPE) | 不变 |

### 4.3 模型规模系列

| 模型 | 层数 | Q头数 / KV头数 | 上下文长度 | Embedding绑定 |
|------|------|---------------|-----------|--------------|
| Qwen2.5-0.5B | 24 | 14 / 2 | 32K | Yes |
| Qwen2.5-1.5B | 28 | 12 / 2 | 32K | Yes |
| Qwen2.5-3B | 36 | 16 / 2 | 32K | Yes |
| Qwen2.5-7B | 28 | 28 / 4 | 128K | No |
| Qwen2.5-14B | 48 | 40 / 8 | 128K | No |
| Qwen2.5-32B | 64 | 40 / 8 | 128K | No |
| Qwen2.5-72B | 80 | 64 / 8 | 128K | No |
| Qwen2.5-Turbo (MoE) | - | - | 128K (1M扩展) | - |
| Qwen2.5-Plus (MoE) | - | - | 128K (1M扩展) | - |

**新增规模**：3B、14B、32B（相比Qwen2新增）。Turbo和Plus为API-only的MoE模型。

### 4.4 训练策略

**预训练**：
- 数据规模从Qwen2的7T tokens大幅扩展至**18T tokens**
- 进一步提升数据质量和多样性
- 加强常识知识、专家知识和推理能力的数据比重

**后训练**：
- SFT：超过**100万样本**的精细化监督微调（相比Qwen2的50万翻倍）
- **多阶段强化学习**（multistage RL）：
  - 改善人类偏好对齐
  - 显著提升长文本生成能力
  - 增强结构化数据分析能力
  - 改善指令遵循质量

### 4.5 关键创新

- **数据规模再飞跃**：从7T到18T tokens，预训练数据质量和规模均大幅提升
- **新增模型规模**：3B和32B填补了模型大小空白，14B回归
- **128K原生上下文**：7B及以上模型原生支持128K上下文
- **1M上下文扩展**：Qwen2.5-Turbo/Plus通过专门技术支持百万token上下文
- **后训练革新**：100万样本SFT + 多阶段RL大幅提升模型实用性
- **专业化衍生**：作为基座训练Qwen2.5-Math、Qwen2.5-Coder、QwQ等专业模型

---

## 5. Qwen3（2025年4月）

### 5.1 发布背景

Qwen3于2025年4月发布，技术报告于2025年5月在arXiv发布（arXiv:2505.09388）。这是Qwen系列最大的架构和训练方法论升级，引入了thinking/non-thinking双模式和全新的MoE设计。

### 5.2 模型架构

| 组件 | 技术方案 | 与Qwen2.5差异 |
|------|----------|--------------|
| 注意力机制 | GQA | 保持 |
| 位置编码 | RoPE (ABF技术, base freq 1,000,000) | 采用ABF技术 |
| 长上下文 | YARN + DCA | 保持，支持4倍推理扩展 |
| FFN激活函数 | SwiGLU | 不变 |
| 归一化 | RMSNorm + Pre-Normalization | 不变 |
| 注意力偏置 | **移除QKV Bias** | 重大变更 |
| 注意力稳定 | **引入QK-Norm** | 新增，替代QKV bias |
| 词表大小 | 151,669 (BBPE) | 微调 |
| MoE设计 | 128专家/8激活，无共享专家 | 移除共享专家 |
| 负载均衡 | Global-batch load balancing loss | 新的均衡策略 |

### 5.3 模型规模系列

#### Dense模型

| 模型 | 层数 | Q头数 / KV头数 | Embedding绑定 | 上下文长度 |
|------|------|---------------|--------------|-----------|
| Qwen3-0.6B | 28 | 16 / 8 | Yes | 32K |
| Qwen3-1.7B | 28 | 16 / 8 | Yes | 32K |
| Qwen3-4B | 36 | 32 / 8 | Yes | 128K |
| Qwen3-8B | 36 | 32 / 8 | No | 128K |
| Qwen3-14B | 40 | 40 / 8 | No | 128K |
| Qwen3-32B | 64 | 64 / 8 | No | 128K |

#### MoE模型

| 模型 | 层数 | Q头数 / KV头数 | 专家 (总/激活) | 上下文长度 |
|------|------|---------------|---------------|-----------|
| Qwen3-30B-A3B | 48 | 32 / 4 | 128 / 8 | 128K |
| Qwen3-235B-A22B | 94 | 64 / 4 | 128 / 8 | 128K |

### 5.4 训练策略

**预训练（三阶段）**：

| 阶段 | 内容 | 数据量 | 序列长度 |
|------|------|--------|----------|
| S1: 通用阶段 | 语言能力和通用知识 | 30T+ tokens | 4,096 |
| S2: 推理阶段 | STEM、代码、推理增强 | ~5T tokens | 4,096 |
| S3: 长上下文阶段 | 长文本数据，扩展上下文 | 数千亿tokens | 32,768 |

- 总训练数据：约**36T tokens**（是Qwen2.5的两倍）
- 覆盖**119种语言和方言**（是Qwen2.5语言数的3倍）
- 使用Qwen2.5-VL从PDF提取文本，Qwen2.5-Math/Coder生成合成数据
- 多维度数据标注系统（教育价值、领域、安全等），实例级数据配比优化

**后训练（四阶段流水线）**：

| 阶段 | 方法 | 目标 |
|------|------|------|
| 1. Long-CoT Cold Start | SFT (验证过的推理数据) | 赋予基础推理能力 |
| 2. Reasoning RL | **GRPO** (3,995组query-verifier对) | 扩展推理探索与利用能力 |
| 3. Thinking Mode Fusion | Continual SFT | 将non-thinking能力融入thinking模型 |
| 4. General RL | RL (20+通用任务) | 强化通用能力，纠正不良行为 |

**GRPO细节**：
- 使用rule-based rewards进行验证
- 大batch size + 高rollout数 + off-policy训练
- 通过控制模型entropy保持训练稳定
- Qwen3-235B-A22B的AIME'24分数从70.1提升至85.1（170步RL训练）

### 5.5 Thinking/Non-Thinking双模式

这是Qwen3最核心的创新：

- **Thinking Mode**：模型在给出最终答案前进行逐步推理（Chain-of-Thought），适合复杂问题
- **Non-Thinking Mode**：模型快速直接响应，适合简单问题
- **统一框架**：两种模式集成在同一个模型中，通过chat template中的 `/think` 和 `/no_think` 控制
- **Thinking Budget**：支持精细控制思考token的预算，性能随计算预算平滑提升
- **实现方式**：通过`enable_thinking=True/False`参数切换

### 5.6 Strong-to-Weak蒸馏

- 直接从大模型教师蒸馏output logits到小模型学生
- 免去对每个小模型执行完整四阶段训练
- 仅需四阶段方法1/10的GPU时间
- 同时提升Pass@1（即时性能）和Pass@64（探索能力）

### 5.7 关键创新总结

- **Thinking/Non-Thinking统一模型**：业界首个将推理模式和快速响应模式融合的开源大模型
- **QK-Norm替代QKV Bias**：更稳定的训练，支持更大规模
- **移除共享专家**：MoE架构简化，配合global-batch balance loss
- **36T tokens预训练**：数据规模再翻倍
- **119种语言**：全球化覆盖能力大幅提升
- **GRPO强化学习**：基于规则验证的高效RL训练
- **Strong-to-Weak蒸馏**：高效的小模型训练方法

---

## 6. Qwen3.5 / Qwen3.6 / Qwen3.7（2025年9月—2026年6月）—— Agent原生时代

### 6.1 概览

2025年4月Qwen3发布后，Qwen团队进入了极其密集的发布期，模型版本快速迭代，核心方向从Thinking推理转向Agent原生设计。

### 6.2 Qwen3-Next-80B-A3B（2025年9月）—— 架构革新

这是后续Qwen3.5/3.6/3.7的技术基础，引入了革命性架构改进：

| 组件 | Qwen3 | Qwen3-Next |
|------|-------|------------|
| 注意力 | 标准GQA | **Hybrid Attention**（Gated DeltaNet + Gated Attention） |
| MoE | 128专家 | 更高稀疏度MoE |
| Token预测 | 单token | **Native MTP**（多token预测原生预训练） |
| 规模 | 235B/22B激活 | 80B/3B激活 |
| 上下文 | 32K-128K | 超长序列优化 |

**Gated DeltaNet**是一种线性注意力变体，与传统 Full Attention相比，在长序列输入下推理开销大幅降低。与Gated Attention混合使用，兼顾了表达能力和效率。

### 6.3 Qwen3.5（2026年2月16日）—— MoE效率革命

| 特征 | 参数 |
|------|------|
| 总参数量 | 397B |
| 激活参数 | 17B |
| MoE配置 | 256路由专家 + 8激活 + 1共享 |
| 原生上下文 | 262K tokens |
| 语言支持 | 201种语言/方言 |
| 吞吐提升 | 8.6x（32K）/ 19x（256K）vs Qwen3-Max |
| 新技术 | Gated Delta Networks |
| 许可证 | Apache 2.0 |

定位为"Towards Native Multimodal Agents"，是Qwen团队"Agent原生"路线的首个重大产物。尺寸系列包括0.8B、9B等多个规模，其中Qwen3.5-9B在多个基准上击败了更大规模的模型。

### 6.4 Qwen3.6 系列（2026年5月）—— 开源新基线

Qwen3.6提供了完整的开源模型矩阵：

| 模型 | 类型 | 特点 |
|------|------|------|
| Qwen3.6-27B | Dense | 单A100可部署，实用工作马 |
| Qwen3.6-72B | Dense | MMLU 88.5%, HumanEval 92.1% |
| Qwen3.6-35B-A3B | MoE | 3B激活，128K上下文（YaRN） |
| Qwen3.6-Plus | API | 生产级托管版本 |
| Qwen3.6-Flash | API | 低延迟优化 |
| Qwen3.6-Max-Preview | API | 实验性前沿能力 |

架构特点：
- MoE + YaRN 128K上下文
- R1-Zero RLHF对齐
- AITemplate内核融合，推理吞吐比Qwen2.5提升2x
- Apache 2.0完全开源

### 6.5 Qwen3.7-Max（2026年5月19日）—— Agent Frontier

Qwen3.7-Max是Qwen团队的闭源旗舰，专为长程自主Agent设计：

| 特征 | 参数 |
|------|------|
| 上下文 | 1M tokens |
| 最大输出 | 65,536 tokens |
| API兼容 | OpenAI spec + Anthropic spec |
| 自主执行 | 35小时连续运行记录 |
| 开源 | 否（仅API） |

关键基准（vs Claude Opus 4.6）：
- Terminal Bench 2.0: 69.7 vs 65.4
- SWE-Verified: 80.4 vs 80.8
- GPQA Diamond: 92.4 vs 91.3
- IMOAnswerBench: 90.0 vs 75.3
- LiveCodeBench: 91.6 vs 88.8
- HMMT 2026 Feb: 97.1 vs 96.2

### 6.6 Qwen3-Coder系列

| 模型 | 时间 | 规模 | 特点 |
|------|------|------|------|
| Qwen3-Coder 480B-A35B | 2025.07 | 480B/35B激活 | 7.5T tokens（70%代码），1M上下文，SWE-Bench SOTA |
| Qwen3-Coder-Next | 2026.05 | 80B/3B激活 | 混合注意力，256K上下文，本地编码Agent，Sonnet 4.5级 |

---

## 7. 跨代对比总结

### 6.1 架构演进对比

| 特性 | Qwen (2023) | Qwen1.5 (2024.2) | Qwen2 (2024.6) | Qwen2.5 (2024.9) | Qwen3 (2025.4) |
|------|-------------|-------------------|-----------------|-------------------|-----------------|
| 注意力机制 | MHA | MHA (110B: GQA) | GQA (全部) | GQA (全部) | GQA (全部) |
| 位置编码 | RoPE | RoPE | RoPE (freq↑) | RoPE | RoPE + ABF |
| FFN | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| 归一化 | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| QKV Bias | Yes | Yes | Yes | Yes | **No (QK-Norm)** |
| 词表大小 | 151,936 | 151,936 | 151,646 | 151,646 | 151,669 |
| Embedding绑定 | 否 | 否 | 小模型是/大模型否 | 小模型是/大模型否 | 小模型是/大模型否 |
| 最大上下文 | 8K→32K | 32K | 32K→128K | 128K (1M) | 128K (推理4x) |
| MoE | 无 | MoE-A2.7B | 57B-A14B | Turbo/Plus (API) | 30B-A3B, 235B-A22B |

### 6.2 训练规模演进

| 代次 | 预训练数据量 | 语言数 | SFT样本数 | 后训练方法 |
|------|-------------|--------|----------|-----------|
| Qwen | ~3T | 中英为主 | - | SFT + RLHF (PPO) |
| Qwen1.5 | ~3T | 12+ | - | SFT + DPO + PPO |
| Qwen2 | 7T (0.5B: 12T) | ~30 | 50万+ | SFT + DPO + Online DPO |
| Qwen2.5 | 18T | ~30 | 100万+ | SFT + 多阶段RL |
| Qwen3 | 36T | 119 | - | CoT SFT + GRPO + Mode Fusion + General RL |

### 6.3 模型规模演进

| 代次 | Dense模型规模 | MoE模型 |
|------|-------------|---------|
| Qwen | 1.8B, 7B, 14B, 72B | 无 |
| Qwen1.5 | 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, 110B | MoE-A2.7B |
| Qwen2 | 0.5B, 1.5B, 7B, 72B | 57B-A14B |
| Qwen2.5 | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | Turbo, Plus (API) |
| Qwen3 | 0.6B, 1.7B, 4B, 8B, 14B, 32B | 30B-A3B, 235B-A22B |

### 6.4 核心突破时间线

```
2023.08  Qwen初代    → 大词表BBPE + QKV Bias + 中英双语基座
2024.02  Qwen1.5     → 统一32K上下文 + 首个MoE + HF transformers集成
2024.06  Qwen2       → GQA全面应用 + 7T数据 + DCA/YARN长上下文 + MoE架构
2024.09  Qwen2.5     → 18T数据 + 128K原生 + 百万SFT + 多阶段RL
2025.04  Qwen3       → Thinking/Non-Thinking双模式 + 36T数据 + GRPO + QK-Norm
```

### 6.5 关键技术趋势

1. **数据scaling**：3T → 3T → 7T → 18T → 36T，每代约翻倍
2. **上下文长度**：8K → 32K → 128K → 262K → 1M，持续突破
3. **注意力效率**：MHA → GQA → GQA + QK-Norm → **Hybrid Attention (GDN + Gated Attn)**
4. **对齐方法**：PPO → DPO → Online DPO → GRPO → R1-Zero
5. **MoE趋势**：从无到有，专家数从8增至256，激活参数比从1/10降至1/23
6. **推理能力**：从通用对话 → Thinking Mode → **长程自主Agent**
7. **设计范式**：从对话式助手 → Agent原生基座模型

---

## 参考文献

| 论文标题 | arXiv ID | 发布时间 |
|----------|----------|----------|
| Qwen Technical Report | arXiv:2309.16609 | 2023年9月 |
| Qwen2 Technical Report | arXiv:2407.10671 | 2024年7月 |
| Qwen2.5 Technical Report | arXiv:2412.15115 | 2024年12月 |
| Qwen3 Technical Report | arXiv:2505.09388 | 2025年5月 |
| Qwen3-VL Technical Report | arXiv:2511.21631 | 2025年11月 |
| Qwen3.5-Omni Technical Report | arXiv:2604.15804 | 2026年4月 |

**官方资源**：
- Qwen1.5 发布博客: https://qwenlm.github.io/blog/qwen1.5/
- Qwen1.5-110B 博客: https://qwen.ai/blog?id=qwen1.5-110b
- Qwen3 发布博客: https://qwenlm.github.io/blog/qwen3/
- Qwen3.5 发布博客: https://qwen.ai/blog?id=qwen3.5
- Qwen3.7 The Agent Frontier: https://qwen.ai/blog?id=qwen3.7
- Qwen3-Coder-Next: https://qwen.ai/blog?id=qwen3-coder-next
- GitHub: https://github.com/QwenLM/Qwen3.6
- Hugging Face: https://huggingface.co/Qwen
