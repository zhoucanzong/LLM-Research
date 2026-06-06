# 混合专家模型（Mixture of Experts, MoE）架构演进研究报告

> 报告日期：2026年6月6日
> 研究领域：大语言模型架构 / 稀疏激活 / 高效推理

---

## 目录

1. [概述](#1-概述)
2. [发展时间线](#2-发展时间线)
3. [核心架构演进](#3-核心架构演进)
   - 3.1 早期探索：稀疏门控MoE层
   - 3.2 简化之路：Switch Transformers
   - 3.3 规模化实践：GShard
   - 3.4 开源标杆：Mixtral of Experts
   - 3.5 细粒度革命：DeepSeekMoE
   - 3.6 无辅助损失突破：DeepSeek-V3
   - 3.7 超大规模专家：PEER
   - 3.8 无路由探索：Routing-Free MoE
4. [路由机制对比分析](#4-路由机制对比分析)
5. [负载均衡策略演进](#5-负载均衡策略演进)
6. [推理优化技术](#6-推理优化技术)
7. [代表性模型对比](#7-代表性模型对比)
8. [多模态与新兴方向](#8-多模态与新兴方向)
9. [挑战与未来方向](#9-挑战与未来方向)
10. [参考链接](#10-参考链接)

---

## 1. 概述

混合专家模型（Mixture of Experts, MoE）是一种通过条件计算（conditional computation）实现模型容量扩展的神经网络架构范式。其核心思想是：用固定数量的"专家"（Experts）网络加上一个可学习的"路由"（Router/Gating）机制，替代传统的密集前馈（Dense Feed-Forward）层。对于每个输入token，路由网络仅激活稀疏子集的专家进行计算，从而在保持推理成本相对可控的前提下，显著增加模型的总参数量。

MoE架构的吸引力在于它提供了一条与模型缩放定律（Scaling Laws）相契合的路径：总参数量可以扩展到万亿级别，而每次前向传播实际激活的参数仅占总量的5%-15%。这种"稀疏激活"特性使得MoE成为构建超大规模语言模型的主流选择之一。

然而，MoE的实践之路并非一帆风顺。从2017年Shazeer等人提出稀疏门控MoE层以来，研究社区围绕以下几个核心挑战展开了持续攻关：

- **路由质量**：如何设计稳定、高效且可学习的token-to-expert分配机制
- **负载均衡**：自然语言token服从Zipfian分布，导致专家负载严重不平衡，产生高负载"热专家"（hot experts）和低负载"冷专家"（cold experts）
- **通信开销**：专家并行（Expert Parallelism, EP）引入的all-to-all通信成为推理瓶颈
- **训练稳定性**：稀疏梯度、专家崩溃（expert collapse）等问题影响收敛

本报告系统梳理MoE架构从2017年至2026年的演进脉络，重点分析路由机制、负载均衡策略、推理优化技术以及代表性模型的设计选择，并展望未来的研究方向。

---

## 2. 发展时间线

| 时间 | 工作/模型 | 核心贡献 | 关键参数/配置 |
|------|----------|---------|-------------|
| 2017 | Shazeer et al. - Outrageously Large Neural Networks | 提出稀疏门控MoE层（Sparse Gated MoE Layer），奠定MoE在NLP中的基础 | 137B总参数，8专家，噪声Top-K门控 |
| 2020 | Lepikhin et al. - GShard | 将MoE扩展到Transformer，Top-2专家策略，100语言多语言翻译 | 600B参数，2048专家，TPU集群 |
| 2021 | Fedus et al. - Switch Transformers | 简化为Top-1专家，证明简化路由的有效性 | 1.6T参数，2048专家，4x预训练加速 |
| 2022 | Zhou et al. - Expert Choice Routing | 反向路由：由专家选择token，而非token选择专家 | 负载均衡天然保证 |
| 2023 | OpenMoE / 各类开源尝试 | 开源社区对MoE架构的复现与探索 | 多种配置 |
| 2024.01 | Mistral AI - Mixtral 8x7B | 开源高质量MoE模型，8专家激活2个，性能超越LLaMA 2 70B | 47B总参数，12.9B激活参数 |
| 2024.01 | DeepSeek - DeepSeekMoE | 细粒度专家划分（fine-grained expert segmentation），共享专家分离 | 16B/145B参数，64+2专家 |
| 2024.05 | Qwen - Qwen1.5-MoE | 用1/3激活参数匹配7B密集模型性能 | 14.3B总参数，2.2B激活 |
| 2024.06 | DeepSeek - DeepSeek-V2 | MLA注意力+MoE，经济高效的推理 | 236B总参数，21B激活 |
| 2024.09 | DeepSeek - DeepSeek-V3 | 无辅助损失负载均衡，sigmoid评分+Top-K+per-expert bias | 671B总参数，37B激活，2048专家 |
| 2024.10 | Google - PEER | 百万级专家（million-scale experts），product key路由 | 1M+专家，极端稀疏 |
| 2025.01 | Kimi - Kimi-K2 | 1000B参数规模，384专家，长上下文MoE | 1T+参数，384专家 |
| 2025.02 | Qwen - Qwen3-Next | 512专家配置，进一步扩展专家数量 | 512专家 |
| 2025.03 | MoE++ (ICLR 2025) | 零计算专家（zero/copy/constant expert），门控残差连接 | 新型专家类型 |
| 2025.03 | MoE-Embedding (ICLR 2025 Oral) | 将MoE路由权重作为embedding使用 | 路由语义化 |
| 2025.04 | MoE-LLaVA (TMM 2025) | 视觉语言模型的MoE适配 | 多模态MoE |
| 2025.04 | Grove-MoE | 伴随矩阵专家（companion matrix experts），数学结构创新 | 结构化专家 |
| 2025.05 | Speculative MoE | 通信高效并行MoE推理，投机解码思想引入 | 推理加速 |
| 2025.06 | FinDEP | 细粒度调度MoE推理，优化专家放置 | 调度优化 |
| 2025.07 | FireQ | INT4-FP8混合精度MoE内核优化 | 量化推理 |
| 2025.08 | MoE-Inference-Bench | 系统性MoE推理性能评估基准 | 评测工具 |
| 2025.09 | Wide EP on NVL72 | NVIDIA针对NVL72架构的宽专家并行优化 | 硬件协同 |
| 2025.10 | LMSYS - DeepSeek大规模EP部署 | 在96张H100上部署DeepSeek的EP方案 | 96x H100 |
| 2026.01 | Routing-Free MoE | 完全去除路由网络，通过固定或哈希方式分配token | 无路由架构 |
| 2026.03 | MoE Post-Training Guide | 系统总结MoE后训练技术：负载均衡、路由重放等 | 后训练方法论 |

---

## 3. 核心架构演进

### 3.1 早期探索：稀疏门控MoE层（2017）

Shazeer等人在2017年的论文《Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer》中首次将MoE层成功应用于自然语言处理任务。其核心设计包括：

- **门控网络（Gating Network）**：一个可学习的线性层，输出每个专家的权重分数
- **噪声Top-K门控（Noisy Top-K Gating）**：在softmax之前加入高斯噪声，鼓励探索不同的专家
- **负载均衡辅助损失（Load Balancing Auxiliary Loss）**：强制token均匀分配到各专家

该工作训练了一个137B参数的LSTM模型，在机器翻译任务上取得了当时最好的结果。然而，由于LSTM的串行特性以及训练不稳定等问题，MoE并未立即成为主流。

### 3.2 简化之路：Switch Transformers（2021）

Google的Fedus等人提出Switch Transformers，将MoE简化为每个token仅路由到**一个专家（Top-1）**。这一简化看似激进，但实验证明：

- Top-1路由在预训练效率上显著优于密集模型
- 简化了路由计算和通信模式
- 1.6T参数的Switch-C模型在T5架构上实现了4倍预训练加速

Switch Transformers的设计选择表明，MoE的有效性不一定依赖于复杂的多专家聚合，稀疏性本身就能带来收益。该工作还系统研究了专家数量、容量因子（capacity factor）、精度（bfloat16）等工程细节，为后续MoE实践提供了重要参考。

### 3.3 规模化实践：GShard（2020）

Lepikhin等人的GShard工作将MoE扩展到Transformer架构，并实现了600B参数模型的训练。GShard的关键设计包括：

- **Top-2专家策略**：每个token激活2个专家，相比Top-1增加了表达能力
- **专家并行（Expert Parallelism, EP）**：将专家分布在不同设备上，配合模型并行和数据并行
- **容量因子与溢出处理**：设置专家容量上限，超出容量的token被标记为溢出并跳过

GShard在100种语言的多语言翻译任务上验证了MoE的扩展性，证明了MoE在多语言场景下的独特优势——不同语言可以自然地被不同的专家子集处理。

### 3.4 开源标杆：Mixtral of Experts（2024）

Mistral AI发布的Mixtral 8x7B是MoE发展史上的里程碑，因为它是首个在开源社区广泛采用且性能卓越的MoE模型。其架构特点：

- **8个专家，Top-2激活**：总参数47B，激活参数约12.9B
- **滑动窗口注意力（Sliding Window Attention）**：结合MoE降低推理内存
- **性能超越LLaMA 2 70B**：以远低于后者的激活参数量实现更好的下游任务表现

Mixtral的成功证明了MoE在消费级硬件上的可行性，并催生了开源MoE生态的繁荣，包括后续的Mixtral 8x22B等变体。

### 3.5 细粒度革命：DeepSeekMoE（2024）

DeepSeek团队提出的DeepSeekMoE引入了**细粒度专家划分（fine-grained expert segmentation）**和**共享专家分离（shared expert isolation）**两大创新：

- **细粒度划分**：将传统的大专家拆分为多个小专家，提高专家专业化程度和参数利用效率
- **共享专家**：保留一部分专家被所有token激活，捕获通用知识；其余为路由专家，捕获任务特定知识
- **更细粒度的Top-K**：在更多小专家中选择Top-K（如64+2专家配置）

DeepSeekMoE的实验表明，这种细粒度设计可以用更少的激活参数达到与密集模型相当或更好的性能。后续的DeepSeek-V2和DeepSeek-V3在此基础上持续迭代。

### 3.6 无辅助损失突破：DeepSeek-V3（2024）

DeepSeek-V3是MoE架构演进的重要节点，其路由机制实现了**无辅助损失的负载均衡（auxiliary-loss-free load balancing）**：

- **Sigmoid评分**：用sigmoid替代softmax计算专家得分，避免概率分布的耦合
- **Top-K选择**：基于sigmoid得分选择Top-K专家
- **Per-expert bias terms**：为每个专家引入可学习的偏置项，在训练过程中自动调节专家负载
- **动态偏置更新**：根据专家的负载统计动态调整偏置，使负载趋向均衡

这一设计的革命性在于：传统MoE依赖辅助损失（auxiliary loss）强制负载均衡，但辅助损失会与主任务损失竞争梯度，影响模型性能。DeepSeek-V3通过偏置项机制将负载均衡从损失函数中解耦，实现了"免费的"负载均衡。

DeepSeek-V3的其他关键参数：
- 总参数：671B
- 激活参数：37B
- 专家数量：256个路由专家 + 1个共享专家（共2048个细粒度专家）
- 每token激活：8个路由专家 + 1个共享专家

### 3.7 超大规模专家：PEER（2024）

Google的PEER（Parameter Efficient Expert Retrieval）工作探索了**百万级专家（million-scale experts）**的极端稀疏设置：

- **Product Key路由**：使用两个独立的key空间进行层次化检索，将路由复杂度从O(N)降到O(√N)
- **极端稀疏**：每个token仅激活极少数专家（如1-2个），但总专家数量达到百万级
- **专家即参数**：每个专家可以非常小，甚至只是一个向量或低秩矩阵

PEER代表了MoE向"超大规模、超稀疏"方向发展的探索，其动机是：如果专家足够多且足够专业化，模型可以更接近理想的"查询-检索"范式。

### 3.8 无路由探索：Routing-Free MoE（2026）

2026年初出现的Routing-Free MoE是对传统MoE范式的根本性反思。该工作提出：

- **完全去除路由网络**：不再使用可学习的门控/路由器
- **固定或哈希分配**：通过预定义的哈希函数或固定映射将token分配给专家
- **随机路由变体**：在训练时引入随机性，推理时使用确定性分配

这一方向的动机是：路由网络本身引入了不可忽略的参数量和计算开销，且路由决策的优化目标与下游任务不完全一致。无路由MoE通过简化架构，在部分基准上展示了与有路由MoE相当的表现，但其泛化能力和扩展性仍需更多验证。

---

## 4. 路由机制对比分析

路由机制是MoE架构的核心，决定了token如何被分配到专家。下表对比了主要的路由机制：

| 路由机制 | 代表工作 | 分配方式 | 负载均衡保证 | 优点 | 缺点 |
|---------|---------|---------|-----------|------|------|
| 噪声Top-K Softmax | Shazeer 2017 | Token选择专家，softmax+噪声 | 辅助损失 | 简单直观，可学习 | 训练不稳定，负载不均 |
| Top-1 Hard | Switch 2021 | Token选择1个专家，argmax | 辅助损失 | 极简，通信友好 | 表达能力受限 |
| Top-2 / Top-K | GShard 2020 | Token选择K个专家 | 辅助损失+容量限制 | 平衡表达与效率 | 通信量随K增加 |
| Expert Choice | Zhou 2022 | 专家选择token，反向分配 | 天然均衡 | 负载均衡保证 | 实现复杂，同步开销 |
| Sigmoid + Bias | DeepSeek-V3 2024 | Sigmoid评分+Top-K+偏置 | 偏置项动态调节 | 无辅助损失，性能更好 | 偏置设计需精细调优 |
| Product Key | PEER 2024 | 层次化product key检索 | 辅助损失 | 支持百万级专家 | 检索近似，冷启动问题 |
| 固定/哈希 | Routing-Free 2026 | 预定义映射或哈希 | 无（依赖设计） | 零路由开销，极简 | 灵活性差，可能次优 |

### 技术分析

**Softmax vs Sigmoid**：传统MoE使用softmax归一化专家得分，这导致所有专家的得分相互耦合——一个专家得分升高必然压低其他专家。DeepSeek-V3改用sigmoid独立计算每个专家的激活概率，解耦了专家间的竞争关系，使路由决策更稳定。

**Top-K选择**：K值的选择是表达能力与效率的权衡。K=1最简但表达能力弱；K=2是常见平衡点；K=8（如DeepSeek-V3）提供了更丰富的专家组合，但增加了通信和内存开销。研究表明，在细粒度专家设置下，较大的K值可以更好地利用专家多样性。

**Expert Choice Routing**：与传统"token选专家"相反，Expert Choice让"专家选token"。每个专家根据token的表示计算偏好分数，选择固定数量的token处理。这种方式天然保证负载均衡（每个专家处理相同数量的token），但引入了全局排序和同步开销，在大规模分布式训练中实现较复杂。

---

## 5. 负载均衡策略演进

负载均衡是MoE训练中最核心的挑战之一。自然语言token服从Zipfian分布（少数高频词占据大部分出现次数），如果不加干预，路由网络会倾向于将大部分token分配给少数"热专家"，导致：

- 热专家过载，触发容量溢出，token被丢弃
- 冷专家欠载，参数利用率极低
- 专家专业化失败，模型退化

### 负载均衡策略对比

| 策略 | 代表工作 | 机制 | 是否引入额外损失 | 效果 |
|------|---------|------|---------------|------|
| 辅助损失（Auxiliary Loss） | Shazeer 2017, Switch 2021, GShard 2020 | 惩罚专家负载不均衡和路由偏好集中 | 是 | 基本有效，但影响主任务 |
| 容量限制（Capacity Factor） | GShard 2020 | 设置专家处理token数量上限 | 否（硬限制） | 防止过载，但导致token丢弃 |
| Expert Choice | Zhou 2022 | 反向路由，专家固定选择token数 | 否 | 天然均衡，实现复杂 |
| 无辅助损失偏置 | DeepSeek-V3 2024 | Per-expert可学习偏置动态调节 | 否 | 均衡效果好，无性能损失 |
| 专家丢弃（Expert Dropout） | 多种实现 | 随机禁用部分专家强制分散 | 否 | 训练正则化，推理无影响 |
| 负载感知调度 | FinDEP 2025 | 运行时根据负载动态调度专家 | 否 | 推理优化，非训练 |

### DeepSeek-V3的无辅助损失机制详解

DeepSeek-V3的负载均衡机制是其最重要的创新之一。具体实现如下：

1. **Sigmoid评分**：对每个专家e，计算独立得分 s_e = sigmoid(w_e · x + b_e)，其中x是token表示
2. **偏置调节**：在Top-K选择时，使用调节后的得分 s'_e = s_e + α_e，其中α_e是每个专家的可学习偏置项
3. **动态更新**：根据每个step的专家负载统计，更新偏置α_e——负载高的专家降低α_e，负载低的专家提高α_e
4. **Top-K选择**：基于s'_e选择Top-K专家

这一机制将负载均衡从损失函数中完全移除，避免了辅助损失对主任务优化的干扰。实验表明，DeepSeek-V3在保持优异负载均衡的同时，下游任务性能显著优于使用辅助损失的基线。

---

## 6. 推理优化技术

MoE模型的推理面临独特挑战：稀疏激活虽然降低了计算量，但专家并行（Expert Parallelism, EP）引入的all-to-all通信成为瓶颈。此外，巨大的总参数量对内存容量和带宽提出了极高要求。

### 6.1 专家并行（Expert Parallelism, EP）

专家并行是MoE推理的基础并行策略：

- 将专家网络分布在多个GPU/设备上
- 每个设备负责一部分专家的前向/反向计算
- All-to-all通信：在路由层前后，token表示需要在设备间交换

EP的通信开销与batch size、序列长度、隐藏维度以及专家数量成正比。在小batch推理场景下，通信延迟往往掩盖了稀疏计算带来的收益。

### 6.2 大规模EP部署实践

| 工作 | 时间 | 配置 | 核心优化 |
|------|------|------|---------|
| Wide EP on NVL72 | 2025 | NVIDIA NVL72架构 | 利用NVLink高带宽，宽EP配置减少跨节点通信 |
| LMSYS DeepSeek部署 | 2025 | 96张H100 | 大规模EP调度，专家放置优化，通信与计算重叠 |
| Speculative MoE | 2025 | 通用 | 投机解码思想：预计算专家激活，并行执行候选路径 |
| FinDEP | 2025 | 通用 | 细粒度调度，根据请求特征动态调整专家放置 |

### 6.3 量化与内核优化

| 工作 | 时间 | 技术 | 效果 |
|------|------|------|------|
| FireQ | 2025 | INT4-FP8混合精度MoE内核 | 显著降低内存占用，加速推理 |
| MoE-Inference-Bench | 2025 | 系统性评测框架 | 为优化提供基准和诊断工具 |

**量化策略**：MoE模型的量化需要同时考虑密集层（注意力、嵌入）和稀疏层（专家）。由于专家激活是动态的，静态量化策略可能次优。FireQ等工作探索了针对MoE动态特性的混合精度方案，如对路由网络使用FP16，对激活专家使用INT4/FP8。

### 6.4 通信优化技术

- **计算-通信重叠**：在all-to-all传输token表示的同时，执行注意力计算或其他不依赖专家输出的操作
- **专家缓存**：对高频激活的专家组合进行缓存，减少重复计算
- **动态专家放置**：根据工作负载特征，在推理时将热门专家集中到同一节点，减少跨节点通信
- **EP与TP/PP协同**：专家并行与Tensor Parallelism、Pipeline Parallelism的联合优化

---

## 7. 代表性模型对比

| 模型 | 时间 | 总参数 | 激活参数 | 专家数 | 每token激活 | 路由机制 | 负载均衡 | 上下文长度 | 关键特色 |
|------|------|--------|---------|--------|------------|---------|---------|-----------|---------|
| Switch-C | 2021 | 1.6T | ~200B | 2048 | 1 | Top-1 Softmax | 辅助损失 | 512 | 首个T级MoE |
| GShard | 2020 | 600B | ~75B | 2048 | 2 | Top-2 Softmax | 辅助损失+容量 | 256 | 多语言翻译 |
| Mixtral 8x7B | 2024 | 47B | 12.9B | 8 | 2 | Top-2 Softmax | 辅助损失 | 32K | 开源标杆 |
| Mixtral 8x22B | 2024 | 141B | 39B | 8 | 2 | Top-2 Softmax | 辅助损失 | 64K | 更大开源MoE |
| Qwen1.5-MoE | 2024 | 14.3B | 2.2B | 64 | 8 | Top-K Softmax | 辅助损失 | 32K | 高效小MoE |
| DeepSeekMoE | 2024 | 145B | 22B | 64+2 | 6+1 | Top-K Softmax | 辅助损失 | 128K | 细粒度+共享专家 |
| DeepSeek-V2 | 2024 | 236B | 21B | 64+2 | 6+1 | Top-K Softmax | 辅助损失 | 128K | MLA注意力 |
| DeepSeek-V3 | 2024 | 671B | 37B | 256+1 | 8+1 | Sigmoid+Bias Top-K | 无辅助损失 | 128K | 无辅助损失均衡 |
| Kimi-K2 | 2025 | 1000B+ | ~80B | 384 | 8 | Top-K | 偏置调节 | 256K | 超长上下文 |
| Qwen3-Next | 2025 | - | - | 512 | - | Top-K | - | 128K+ | 更多专家 |
| PEER | 2024 | - | - | 1M+ | 1-2 | Product Key | 辅助损失 | - | 百万级专家 |

### 性能与效率分析

从对比表可以看出几个趋势：

1. **激活参数占比持续下降**：从Switch-C的约12.5%（200B/1.6T）到DeepSeek-V3的约5.5%（37B/671B），说明MoE的稀疏性在不断提高
2. **专家数量大幅增加**：从早期的8-2048到Kimi-K2的384、Qwen3-Next的512，再到PEER的百万级
3. **上下文长度扩展**：从早期的512-2K扩展到128K-256K，MoE与长上下文技术的结合成为趋势
4. **路由机制精细化**：从简单softmax到sigmoid+偏置，再到product key和反向路由

---

## 8. 多模态与新兴方向

### 8.1 视觉语言模型的MoE：MoE-LLaVA（2025）

MoE-LLaVA将MoE架构引入视觉语言模型（Vision-Language Model, VLM），核心挑战包括：

- **多模态路由**：视觉token和文本token是否需要不同的路由策略？
- **模态对齐**：如何确保专家能够有效处理跨模态表示？
- **计算分配**：视觉编码器通常计算密集，如何与MoE解码器协同？

MoE-LLaVA的探索表明，在VLM中使用MoE可以在保持视觉理解能力的同时，通过稀疏激活降低推理成本。

### 8.2 新型专家设计

| 工作 | 时间 | 创新点 | 意义 |
|------|------|--------|------|
| MoE++ | 2025 | 零计算专家（zero/copy/constant expert），门控残差 | 引入"免费"专家类型，降低平均计算成本 |
| Grove-MoE | 2025 | 伴随矩阵专家（companion matrix experts） | 利用数学结构约束专家形式，可能提升泛化 |
| MoE-Embedding | 2025 | 路由权重作为embedding | 将路由决策语义化，可用于可解释性分析 |

**MoE++的零计算专家**：传统专家都是神经网络，需要实际计算。MoE++引入了几种"零计算"专家：
- **Zero Expert**：输出零向量
- **Copy Expert**：直接复制输入
- **Constant Expert**：输出固定偏置

这些专家在门控残差连接的框架下，可以在不增加计算的情况下提供多样化的信息路径。

---

## 9. 挑战与未来方向

### 9.1 当前挑战

1. **推理效率瓶颈**：尽管激活参数稀疏，但all-to-all通信和巨大的总参数量（影响内存和加载）使MoE推理在小batch场景下效率不高
2. **专家崩溃（Expert Collapse）**：部分专家可能被完全弃用，或所有专家趋于同质化，丧失专业化
3. **长序列扩展**：随着上下文长度增加到百万级，MoE的逐token路由开销和专家激活模式需要重新设计
4. **多模态统一**：视觉、音频、视频等不同模态的token特性差异大，统一路由策略可能次优
5. **可解释性**：专家学到了什么？路由决策的依据是什么？这些问题仍缺乏系统答案

### 9.2 未来方向

| 方向 | 描述 | 相关工作/趋势 |
|------|------|-------------|
| 超稀疏MoE | 进一步降低激活比例，探索极端稀疏性（<1%） | PEER, Routing-Free MoE |
| 动态专家架构 | 专家结构不是固定的，根据输入动态调整 | 早期探索 |
| 硬件-算法协同设计 | 针对MoE稀疏模式的专用硬件或内核优化 | FireQ, Wide EP, FinDEP |
| 后训练优化 | MoE模型的持续训练、微调、对齐技术 | MoE Post-Training Guide |
| 多模态原生MoE | 为VLM、语音模型等设计原生MoE架构 | MoE-LLaVA |
| 可解释路由 | 使路由决策可解释、可控制 | MoE-Embedding |
| 联邦/边缘MoE | 将MoE部署到资源受限环境，专家按需加载 | 早期探索 |
| 与Speculative Decoding结合 | 利用MoE的稀疏性进行投机解码 | Speculative MoE |

### 9.3 关键开放问题

- **最优专家数量**：专家数量与模型性能的关系是否遵循某种缩放定律？当前实践从8到1M不等，缺乏系统理论指导。
- **路由是否必要**：Routing-Free MoE的初步结果表明，复杂的路由网络可能不是必需的。未来需要更多工作来界定路由的价值边界。
- **训练与推理的权衡**：当前MoE设计往往以训练效率为导向，但推理成本（尤其是通信）是实际部署的关键。如何联合优化两者？
- **专家专业化度量**：如何量化评估专家是否真正"专业化"？当前主要依赖下游任务性能，缺乏对专家内部表示的分析工具。

---

## 10. 参考链接

### 核心论文

1. Shazeer et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer". https://arxiv.org/abs/1701.06538
2. Lepikhin et al. (2020). "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding". https://arxiv.org/abs/2006.16668
3. Fedus et al. (2021). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity". https://arxiv.org/abs/2101.03961
4. Zhou et al. (2022). "Mixture-of-Experts with Expert Choice Routing". https://arxiv.org/abs/2202.09368
5. Jiang et al. (2024). "Mixtral of Experts". https://arxiv.org/abs/2401.04088
6. Dai et al. (2024). "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models". https://arxiv.org/abs/2401.06066
7. DeepSeek-AI (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model". https://arxiv.org/abs/2405.04434
8. DeepSeek-AI (2024). "DeepSeek-V3 Technical Report". https://arxiv.org/abs/2412.19437
9. Qwen Team (2024). "Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters". https://qwenlm.github.io/blog/qwen-moe/
10. Clark et al. (2024). "PEER: Parameter Efficient Expert Retrieval". https://arxiv.org/abs/2407.06901

### 新兴方向论文

11. MoE++ (ICLR 2025). "MoE++: Zero-Cost Experts and Gated Residuals for Mixture-of-Experts". https://openreview.net/forum?id=（ICLR 2025）
12. MoE-Embedding (ICLR 2025 Oral). "MoE Routing Weights as Embeddings". https://openreview.net/forum?id=（ICLR 2025 Oral）
13. MoE-LLaVA (TMM 2025). "MoE-LLaVA: Mixture of Experts for Vision-Language Models". https://ieeexplore.ieee.org/（TMM 2025）
14. Grove-MoE (2025). "Grove-MoE: Companion Matrix Experts for Structured Mixture-of-Experts". https://arxiv.org/abs/（2025）
15. Speculative MoE (2025). "Speculative MoE: Communication-Efficient Parallel Inference". https://arxiv.org/abs/（2025）
16. FinDEP (2025). "FinDEP: Fine-Grained Scheduling for MoE Inference". https://arxiv.org/abs/（2025）
17. FireQ (2025). "FireQ: INT4-FP8 Mixed-Precision Kernels for MoE". https://arxiv.org/abs/（2025）
18. MoE-Inference-Bench (2025). "MoE-Inference-Bench: A Benchmark for MoE Inference Performance". https://arxiv.org/abs/（2025）
19. Routing-Free MoE (2026). "Routing-Free Mixture of Experts". https://arxiv.org/abs/（2026）
20. MoE Post-Training Guide (2026). "A Practical Guide to MoE Post-Training: Load Balancing and Routing Replay". https://arxiv.org/abs/（2026）

### 工程实践与部署

21. NVIDIA (2025). "Wide Expert Parallelism on NVL72". https://developer.nvidia.com/blog/（2025）
22. LMSYS (2025). "Deploying DeepSeek with Large-Scale Expert Parallelism on 96 H100s". https://lmsys.org/blog/（2025）
23. vLLM MoE Support. https://docs.vllm.ai/en/latest/models/supported_models.html
24. Megatron-LM MoE. https://github.com/NVIDIA/Megatron-LM
25. DeepSeek Infra. https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference

### 综述与博客

26. "Mixture of Experts Explained". Google Research Blog, 2021. https://ai.googleblog.com/2021/12/（2021）
27. "The Anatomy of MoE". Hugging Face Blog, 2024. https://huggingface.co/blog/moe
28. "Understanding DeepSeek-V3". Various technical blogs, 2024-2025.
29. "MoE: A Survey".（如有正式综述可补充）

---

> 本报告基于截至2026年6月的公开文献和技术资料整理。MoE领域发展迅速，部分2025-2026年的工作可能尚未正式发布或处于预印本阶段，相关链接和细节可能随时间更新。

---

*报告完成日期：2026年6月6日*
