<div align="center">

# 🤖 LLM Research Reports

**AI Agent 驱动的中文大语言模型深度调研报告集**

> ⚠️ **重要声明**：本知识库所有报告均由 AI Agent 自动调研生成，信息来源于公开网络搜索、第三方评测站点、社交媒体及行业媒体报道。由于大模型厂商官方披露程度不一，部分日期、参数、benchmark 数据可能存在错误、过时或未经证实的情况。**所有内容仅供参考，不构成权威结论，建议读者直接查阅各厂商官方渠道获取最新准确信息。**

覆盖 **16 个主流模型 / 系列 + 1 个技术专题** · 持续更新中

[![Reports](https://img.shields.io/badge/Reports-16-blue?style=flat-square)](.)
[![Files](https://img.shields.io/badge/Files-23-green?style=flat-square)](.)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)

</div>

---

## 📊 总览

| #  | 图标 | 模型 / 系列                                   | 所属公司  | 报告数 | 亮点                                     |
| -- | :--: | --------------------------------------------- | --------- | :----: | ---------------------------------------- |
| 1  | 🇺🇸 | [**Claude**](./Claude-research/)           | Anthropic |   1   | Reasoning、Coding、工具使用能力调研      |
| 2  | 🇨🇳 | [**DeepSeek**](./DeepSeek-research/)       | 深度求索  |   3   | 文本模型 + 多模态 + 专项能力，分拆详述   |
| 3  | 🇨🇳 | [**Doubao-Seed**](./Doubao-Seed-research/) | 字节跳动  |   1   | 豆包大模型 + Seed 系列技术报告           |
| 4  | 🇺🇸 | [**Gemini**](./Gemini-research/)           | Google    |   1   | 多模态理解、长上下文、生态整合           |
| 5  | 🇺🇸 | [**Grok**](./Grok-research/)               | xAI       |   1   | 实时X数据、超大上下文、Musk生态绑定       |
| 6  | 🇨🇳 | [**GLM**](./GLM-research/)                 | 智谱 AI   |   1   | ChatGLM 系列演进与技术路线               |
| 7  | 🇺🇸 | [**GPT**](./GPT-research/)                 | OpenAI    |   1   | GPT-4 系列能力边界与生态                 |
| 8  | 🇨🇳 | [**Hunyuan**](./Hunyuan-research/)         | 腾讯      |   1   | 混元大模型多模态能力调研                 |
| 9  | 🇨🇳 | [**Kimi**](./Kimi-research/)               | 月之暗面  |   1   | 长上下文、联网搜索特色能力               |
| 10 | 🇺🇸 | [**LLaMA**](./LLaMA-research/)             | Meta      |   1   | 开源模型生态与技术架构                   |
| 11 | 🇨🇳 | [**MiMo**](./MiMo-research/)               | 小米      |   1   | 端侧部署与小模型优化方向                 |
| 12 | 🇨🇳 | [**MiniMax**](./MiniMax-research/)         | MiniMax   |   1   | 多模态与语音能力调研                     |
| 13 | 🇫🇷 | [**Mistral**](./Mistral-research/)         | Mistral AI|   1   | 欧洲开源领袖、GDPR主权、多语言Tier-1     |
| 14 | 🇺🇸 | [**Nemotron**](./Nemotron-research/)       | NVIDIA    |   1   | 混合Mamba架构、合成数据、Agentic高效推理 |
| 15 | 🇨🇳 | [**Qwen**](./Qwen-research/)               | 阿里      |   4   | 文本 + VL + Omni + 生态，覆盖面最广      |
| 16 | 🔬 | [**SpeculativeDecoding**](./SpeculativeDecoding-research/) | 学术界/工业界 | 1 | 投机解码技术演进：EAGLE/DFlash/多模态 |

> **🇨🇳 国内 8 家** &nbsp;|&nbsp; **🇺🇸 海外 5 家** &nbsp;|&nbsp; **🇫🇷 欧洲 1 家** &nbsp;|&nbsp; **🔬 技术专题 1 个** &nbsp;|&nbsp; 共 **23 份** Markdown 报告

---

## 📁 仓库结构

```
LLM-research/
├── Claude-research/                # Anthropic Claude 系列
│   └── report.md
├── DeepSeek-research/              # 深度求索 DeepSeek 系列
│   ├── report.md                   #   主报告
│   ├── deepseek-text-models.md     #   文本模型专题
│   └── deepseek-multimodal-...     #   多模态与专项能力
├── Doubao-Seed-research/           # 字节跳动豆包 / Seed
├── GLM-research/                   # 智谱 GLM 系列
├── GPT-research/                   # OpenAI GPT 系列
├── Gemini-research/                # Google Gemini 系列
├── Grok-research/                  # xAI Grok 系列
├── Hunyuan-research/               # 腾讯混元系列
├── Kimi-research/                  # 月之暗面 Kimi 系列
├── LLaMA-research/                 # Meta LLaMA 系列
├── MiMo-research/                  # 小米 MiMo 系列
├── MiniMax-research/               # MiniMax 系列
├── Mistral-research/               # Mistral AI 系列
├── Nemotron-research/              # NVIDIA Nemotron 系列
├── Qwen-research/                  # 阿里通义千问系列
│   ├── report.md                   #   主报告
│   ├── qwen-text-models.md         #   文本模型专题
│   ├── qwen-vl-models.md           #   视觉语言模型专题
│   └── qwen-omni-and-ecosystem.md  #   Omni 模型与生态
└── SpeculativeDecoding-research/     # 投机解码技术专题
    ├── report.md                       #   主报告：概述、时间线、瓶颈、框架、QwenAccel-Train
    ├── speculative-classical-and-self.md   #   经典方法与自投机（Medusa/Hydra/Lookahead）
    ├── speculative-eagle-series.md         #   EAGLE系列（EAGLE-1/2/3/SpecForge）
    ├── speculative-block-diffusion.md        #   块扩散与检索式方法（DFlash/REST/PLD）
    ├── speculative-training-optimization.md #   训练目标优化（LK Losses/PPOW/ConFu）
    ├── speculative-multimodal.md           #   多模态投机解码（VLM/视频/VLA/语音）
    └── speculative-systems-and-frameworks.md #   系统级优化与框架生态（vLLM/SGLang/TensorRT-LLM）
```

---

## 🔍 使用方式

```bash
# 克隆仓库
git clone <repo-url>
cd LLM-research

# 浏览某份报告
cat Qwen-research/report.md

# 全文搜索（例：搜索所有提到 "MoE" 的报告）
grep -r "MoE" --include="*.md" .
```

---

## ✨ 特色

- 🧠 **AI Agent 生成** — 所有报告由 Claude / DeepSeek / Kimi 等 Agent 自动调研撰写
- 📊 **结构化内容** — 统一模板：模型列表 → 关键能力 → 生态 → 竞品对比
- 🌐 **中英双语** — 报告内容以中文为主，关键术语保留英文
- 🔗 **可追溯** — 部分报告附带来源索引、论文原文、阅读笔记

---

## 🤝 贡献

欢迎提交 PR 补充新的模型调研、更新过时内容或修正错误。

---

<div align="center">

*Made with ❤️ by AI Agents & [zhoucanzong](https://github.com/zhoucanzong)*

</div>
