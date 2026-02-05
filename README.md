# SIGUVC
Synonymous language generation under vocabulary constraints - 智谱项目 - 词库约束下的同义语言生成

## ✨ 核心功能

| 功能模块 | 说明 |
|---------|------|
| **词库构建** | 从语料（人民日报等）中通过分词（pkuseg）与实体抽取（LLM）构建词库，支持五级分类、BGE + FAISS 语义向量化检索 |
| **词汇优化选择** | Lazy Greedy + Beam Search 贪心算法最大化语义覆盖度；基于稀疏矩阵的支配剪枝（GPU 加速）；迭代词表优化器 |
| **受限文本生成** | DP 约束解码——在 token 级别硬约束生成内容，确保输出句仅由词表中的词汇构成；支持多步前瞻优化 |
| **多轮迭代重写** | 拆分子句 → 子句重写 → 反馈修正 → 句子合并，每轮检查词汇覆盖率 |
| **LLM 微调训练** | SFT（监督微调）→ DPO（直接偏好优化）→ PPO（近端策略优化）三阶段对齐训练 |
| **多维度评测** | 词汇覆盖率、语义相似度（基于依存句法匹配）、流畅度、修辞手法检测、综合加权得分 |

## 🏗️ 项目结构

```
├── README.md                       # 项目说明文档
│
├── ─── 根目录脚本 ───               # 独立算法 / 端到端流程脚本
│   ├── alg_dominance.py            # 支配剪枝算法（稀疏矩阵 / GPU 加速）
│   ├── alg_greedy.py               # 贪心 / 波束搜索词汇选择算法
│   ├── build_wordbank.py           # 词库构建流水线
│   ├── category_utils.py           # 五级分类词典 + FAISS 分类
│   ├── data_manager.py             # 词 - 语义矩阵加载器
│   ├── my_LogitsProcessor.py       # DP 约束解码核心（LogitsProcessor）
│   ├── simple_generation.py        # 自定义受限生成函数（集成 DP 约束）
│   ├── local_rewrite.py            # 本地模型重写流水线（入口脚本）
│   ├── zai_rewrite.py              # 智谱 AI API 重写流水线（入口脚本）
│   ├── word_classification.py      # 词汇分类流水线
│   ├── words_analysis.py           # 词汇统计分析
│   ├── evaluation.py               # STS / WMT 评测入口
│   ├── gen_dict2context_words_mappings.py  # 词典→上下文词映射生成
│   └── debug.py                    # 端到端调试脚本
│
├── src/                            # 核心模块（训练 / 推理 / 数据 / 评测）
├── local_modules/                  # 本地模型重写模块
├── zai_modules/                    # 智谱 AI API 重写模块
├── configs/                        # LLaMA-Factory 训练配置
└── LLaMA-Factory/                  # 训练框架（第三方，LLaMA-Factory）
   
```
