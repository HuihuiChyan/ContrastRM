# ContrastRM: 对比奖励模型训练与偏见评估框架

ContrastRM 是一个专门用于训练和评估奖励模型的完整框架，特别关注模型在各种偏见类型上的表现。该项目实现了对比学习和成对学习两种训练模式，并提供了全面的偏见评估工具。

## 🚀 主要功能

本项目通过四个核心模块实现完整的奖励模型训练和评估流程：

### 1. 📊 数据构建 (Data Construction)
**脚本**: `run_construction.sh`

自动化生成带有偏见的训练数据，用于训练更加鲁棒的奖励模型。

**主要功能**:
- 使用 LLM API 为原始数据注入各种类型的偏见回答
- 支持 7 种偏见类型：`length`, `authority`, `beauty`, `assertiveness`, `sycophancy`, `sentiment`, `concreteness`
- 自动验证生成回答的正确性和质量
- 支持批量处理和断点续传

**核心代码文件**:
- `codes/construct/gptout.py`: 偏见注入主程序，随机选择偏见类型并生成对应的偏见回答
- `codes/construct/verify_correctness.py`: 验证生成回答的正确性
- `codes/construct/build_prompt.py`: 构建不同偏见类型的提示模板
- `codes/construct/api.py`: LLM API 调用接口

### 2. 🎯 模型训练 (Model Training)
**脚本**: `run_training.sh`

训练对比奖励模型，支持多种训练模式和优化策略。

**主要功能**:
- **对比学习模式 (Contrastive)**: 使用所有 rejected 回答进行对比学习
- **成对学习模式 (Pairwise)**: 使用 margin loss 进行成对比较学习
- 支持 DeepSpeed 分布式训练
- 混合精度训练 (BF16/FP16)

**核心代码文件**:
- `codes/train/train_reward_model.py`: 主训练脚本，实现完整的训练流程
- `codes/train/reward_model.py`: 奖励模型架构定义
- `codes/train/data_utils.py`: 数据加载和预处理工具

**训练模式对比**:
```python
# 对比学习: [rejected_1, rejected_2, ..., chosen] -> CrossEntropyLoss
# 成对学习: max(0, margin - (chosen_score - rejected_score))
```

### 3. 🔍 偏见评估 (Bias Evaluation)
**脚本**: `run_evaluation_rm.sh`

全面评估训练好的奖励模型在各种偏见类型上的表现。

**主要功能**:
- 支持 10 种偏见类型的评估：`length`, `authority`, `beauty`, `assertiveness`, `sycophancy`, `sentiment`, `concreteness`, `gender`, `race`, `refinement-aware`
- 计算偏见敏感率 (Bias Sensitivity Rate)
- 生成详细的评估报告和统计摘要
- 支持批量评估和单个偏见类型评估

**核心代码文件**:
- `codes/evaluate/eval_rm.py`: 偏见评估主程序
- `data/eval/`: 各种偏见类型的评估数据集

**评估指标**:
- **原始数据对准确率**: 模型在无偏见数据上的表现
- **偏见数据对准确率**: 模型在有偏见数据上的表现  
- **偏见敏感率**: 模型受偏见影响的程度

### 4. 📈 基准评估 (Benchmark Evaluation)
**脚本**: `run_evaluation_rewardbench.sh`

在标准 Reward-Bench 数据集上评估模型性能，提供可比较的基准结果。

**主要功能**:
- 支持 Reward-Bench 数据集的四个主要类别：
  - **Chat**: 日常对话场景
  - **Chat Hard**: 困难对话场景
  - **Safety**: 安全性评估
  - **Reasoning**: 推理能力评估
- 自动计算加权平均准确率
- 生成标准化的评估报告

**核心代码文件**:
- `codes/evaluate/eval_reward_bench.py`: Reward-Bench 评估主程序
- `data/eval/reward_bench/`: Reward-Bench 数据加载工具

## 📁 项目结构

```
ContrastRM/
├── codes/                          # 核心代码目录
│   ├── construct/                  # 数据构建模块
│   │   ├── api.py                 # LLM API 接口
│   │   ├── gptout.py              # 偏见注入主程序
│   │   ├── gptinst.py             # 指令生成工具
│   │   ├── verify_correctness.py  # 回答验证工具
│   │   └── build_prompt.py        # 提示模板构建
│   ├── train/                      # 模型训练模块
│   │   ├── train_reward_model.py  # 训练主程序
│   │   ├── reward_model.py        # 奖励模型定义
│   │   └── data_utils.py          # 数据处理工具
│   └── evaluate/                   # 评估模块
│       ├── eval_rm.py             # 偏见评估
│       ├── eval_reward_bench.py   # 基准评估
│       └── api.py                 # 评估 API 接口
├── data/                           # 数据目录
│   ├── train/                     # 训练数据
│   ├── eval/                      # 评估数据
│   └── train-unified-feedback/    # 数据构建过程数据
├── configs/                        # 配置文件
│   ├── ds_z2_config.json         # DeepSpeed ZeRO-2 配置
│   └── ds_z3_config.json         # DeepSpeed ZeRO-3 配置
├── output/                         # 模型输出目录
├── results/                        # 评估结果目录
├── run_construction.sh            # 数据构建脚本
├── run_training.sh               # 模型训练脚本
├── run_evaluation_rm.sh          # 偏见评估脚本
├── run_evaluation_rewardbench.sh # 基准评估脚本
└── requirements.txt              # 依赖包列表
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

**ContrastRM** - 让奖励模型训练更加鲁棒和可靠 🚀