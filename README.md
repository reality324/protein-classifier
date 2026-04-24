# ProteinClassifier

蛋白质功能预测项目 - 支持多种机器学习算法

## 项目结构

```
ProteinClassifier/
├── scripts/
│   ├── train/                    # 训练脚本
│   │   ├── train_rf.py           # RandomForest训练
│   │   ├── train_xgb.py          # XGBoost训练
│   │   ├── train_mlp.py          # MLP神经网络训练
│   │   └── train_multitask.py    # 多任务神经网络训练
│   │
│   ├── inference/                 # 推理脚本
│   │   ├── inference_rf.py       # RF推理
│   │   ├── inference_xgb.py      # XGBoost推理
│   │   ├── inference_mlp.py      # MLP推理
│   │   ├── inference_bnn.py      # BNN推理
│   │   └── inference_multitask.py # 多任务推理
│   │
│   └── utils/                     # 工具脚本
│       ├── compare.py             # 模型对比
│       └── generate_esm2_features.py # ESM2特征生成
│
├── models/                        # 模型文件
│   ├── rf/                        # RandomForest模型
│   │   ├── rf_esm2_model.pkl     # ESM2编码模型
│   │   ├── rf_ctd_model.pkl       # CTD编码模型
│   │   └── rf_onehot_model.pkl    # OneHot编码模型
│   │
│   ├── xgb/                       # XGBoost模型
│   │   └── xgb_esm2_model.pkl
│   ├── mlp/                       # MLP模型
│   │   └── mlp_esm2_model.pt
│   ├── bnn/                       # 贝叶斯神经网络模型
│   │   └── bnn_esm2_model.pt
│   └── multitask/                 # 多任务模型
│       └── multitask_model.pt
│
├── src/
│   ├── encodings/                 # 特征编码
│   │   ├── onehot.py
│   │   ├── ctd.py
│   │   └── esm2.py
│   │
│   ├── algorithms/                # 算法实现
│   │   ├── rf.py
│   │   ├── xgb.py
│   │   ├── mlp.py
│   │   └── bnn.py
│   │
│   └── pipeline/                   # 训练流程
│       ├── dataset.py
│       ├── trainer.py
│       └── evaluator.py
│
└── data/
    ├── datasets/                   # 数据集
    ├── processed/                  # 处理后的数据
    └── raw/                        # 原始数据
```

## 快速开始

### 训练模型

```bash
# RandomForest (ESM2编码)
python scripts/train/train_rf.py --encoding esm2

# XGBoost
python scripts/train/train_xgb.py --encoding esm2

# MLP神经网络
python scripts/train/train_mlp.py --encoding esm2

# 多任务模型
python scripts/train/train_multitask.py
```

### 模型推理

```bash
# RF模型推理
python scripts/inference/inference_rf.py \
    --model models/rf/rf_esm2_model.pkl \
    --sequence "YOUR_PROTEIN_SEQUENCE"

# XGBoost模型推理
python scripts/inference/inference_xgb.py \
    --model models/xgb/xgb_esm2_model.pkl \
    --sequence "YOUR_PROTEIN_SEQUENCE"

# MLP模型推理
python scripts/inference/inference_mlp.py \
    --model models/mlp/mlp_esm2_model.pt \
    --sequence "YOUR_PROTEIN_SEQUENCE"

# BNN模型推理 (带不确定性估计)
python scripts/inference/inference_bnn.py \
    --model models/bnn/bnn_esm2_model.pt \
    --sequence "YOUR_PROTEIN_SEQUENCE"

# 多任务模型推理 (同时预测EC主类+细胞定位+分子功能)
python scripts/inference/inference_multitask.py \
    --model models/multitask/multitask_model.pt \
    --sequence "YOUR_PROTEIN_SEQUENCE"
```

### 批量推理 (FASTA文件)

```bash
python scripts/inference/inference_rf.py \
    --model models/rf/rf_esm2_model.pkl \
    --fasta proteins.fasta \
    --output results.json
```

## 支持的算法

| 算法 | 训练脚本 | 推理脚本 | 模型格式 | 特点 |
|------|---------|---------|----------|------|
| RandomForest | `train_rf.py` | `inference_rf.py` | `.pkl` | 传统机器学习，可解释性强 |
| XGBoost | `train_xgb.py` | `inference_xgb.py` | `.pkl` | 梯度提升，高精度 |
| MLP | `train_mlp.py` | `inference_mlp.py` | `.pt` | 神经网络，全连接层 |
| BNN | `train_bnn.py` | `inference_bnn.py` | `.pt` | 贝叶斯神经网络，可输出预测不确定性 |
| Multitask | `train_multitask.py` | `inference_multitask.py` | `.pt` | 多任务学习，同时预测EC+定位+功能 |

## 支持的编码方式

- **onehot**: 20维 (20种氨基酸)
- **ctd**: 147维 (氨基酸组成、转换、分布)
- **esm2**: 320维 (ESM2蛋白质语言模型)

## 模型配置

每个模型目录下包含:
- `*_model.*`: 模型文件
- `*_config.json`: 模型配置

```json
{
  "algorithm": "RandomForest",
  "encoding": "esm2",
  "input_dim": 320,
  "n_classes": 7,
  "test_accuracy": 0.85,
  "test_f1_macro": 0.82
}
```
