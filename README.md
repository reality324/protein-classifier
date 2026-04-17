# ProteinClassifier

Multi-task Protein Classifier - Predicting EC Numbers, Cellular Localization, and Function

## Project Structure

```
ProteinClassifier/
├── data/
│   ├── datasets/                  # Raw datasets (parquet format)
│   │   ├── train.parquet          # Training set (~425K samples)
│   │   ├── val.parquet            # Validation set (~53K samples)
│   │   ├── test.parquet           # Test set (~53K samples)
│   │   └── label_mapping.json      # Label mappings
│   └── processed/
│       └── esm2_balanced/         # Balanced sampling dataset (8K train/4K val/4K test)
│           ├── train_features.npy
│           ├── val_features.npy
│           ├── test_features.npy
│           └── train_labels.npy
│
├── models/                       # Trained models (8K ESM2 balanced dataset)
│   ├── mlp_multitask.pt          # MLP model (89.12% EC accuracy)
│   └── bnn_multitask.pt          # BNN model
│
├── experiments/
│   ├── MLP/                      # MLP experiment results
│   ├── BNN/                      # BNN experiment results
│   ├── MultiTask/                # Multi-task model
│   ├── TraditionalML/            # Traditional ML experiments
│   └── plots/                    # Performance comparison plots
│       ├── model_comparison.png
│       └── confusion_matrix.png
│
├── scripts/
│   ├── data/
│   │   └── prepare_data.py       # Data preparation
│   ├── features/
│   │   └── extract_esm2_balanced.py  # ESM2 feature extraction
│   ├── experiments/
│   │   ├── compare_models.py      # Model comparison
│   │   ├── train_multitask.py    # Multi-task training
│   │   └── plot_comparison.py    # Plot comparison
│   ├── evaluation/
│   │   └── compare_algorithms.py  # Algorithm comparison
│   ├── train.py                  # Training entry point
│   ├── predict.py                # Prediction entry point
│   └── api_service.py            # API service
│
└── README.md
```

## Model Performance Comparison

### EC Number Classification (8 classes: 0-7 first digit)

| Model | Accuracy | F1 (micro) | F1 (macro) | Notes |
|-------|----------|------------|------------|-------|
| **MLP** | **89.60%** | 0.8960 | 0.8958 | Best performance |
| **LightGBM** | **89.60%** | 0.8960 | 0.8966 | Tied best |
| XGBoost | 88.98% | 0.8898 | 0.8902 | Stable |
| BNN | 88.72% | 0.8872 | 0.8867 | Needs tuning |

### Per-Class F1 Scores

| Class | MLP | BNN | XGBoost | LightGBM |
|-------|-----|-----|---------|----------|
| No EC | 0.81 | 0.79 | 0.80 | 0.81 |
| EC1-Oxidoreductases | 0.90 | 0.91 | 0.89 | 0.90 |
| EC2-Transferases | 0.85 | 0.84 | 0.82 | 0.84 |
| EC3-Hydrolases | 0.86 | 0.85 | 0.87 | 0.86 |
| EC4-Lyases | 0.89 | 0.88 | 0.89 | 0.89 |
| EC5-Isomerases | 0.92 | 0.90 | 0.92 | **0.93** |
| EC6-Ligases | **0.98** | 0.97 | 0.96 | 0.97 |
| EC7-Translocases | 0.96 | 0.96 | 0.97 | **0.98** |

### Multi-task Model Performance (8K ESM2 Balanced Dataset)

| Task | Accuracy/F1 (micro) | F1 (macro) | Classes |
|------|---------------------|------------|---------|
| EC Classification | **89.12%** | - | 8 (EC 0-7) |
| Localization Classification | 78.44% | 48.05% | 11 |
| Function Classification | 86.43% | 61.74% | 17 |

## Quick Start

### 1. Data Preparation

```bash
cd /home/path/to/your/...
python scripts/data/prepare_data.py
```

### 2. Feature Extraction (if re-extraction needed)

```bash
python scripts/features/extract_esm2_balanced.py
```

### 3. Model Training

```bash
# Model comparison (MLP, BNN, XGBoost, LightGBM)
python scripts/experiments/compare_models.py

# Multi-task training (EC + Localization + Function)
python scripts/experiments/train_multitask.py
```

### 4. Prediction

```bash
python scripts/predict.py
```

## Data Description

### Raw Data

- Source: UniProt SwissProt
- Total samples: ~570K proteins
- Labeled: ~350K entries

### Balanced Dataset (ESM2-35M)

| Split | Samples | Notes |
|-------|---------|-------|
| Training set | 8,000 | 1,000 per EC class (0-7) |
| Validation set | 4,000 | 500 per EC class |
| Test set | 4,000 | 500 per EC class |
| Features | 480-dim | ESM2-35M embeddings |

### Label Classes

| Task | Classes | Description |
|------|---------|-------------|
| EC Number | 8 classes | First digit (0-7) |
| Cellular Localization | 11 classes | Multi-label classification |
| Molecular Function | 17 classes | Multi-label classification |

## Feature Extraction

- **Model**: ESM-2 (facebook/esm2_t6_8M_UR50D)
- **Source**: HuggingFace auto-download
- **Dimension**: 480
- **Max Sequence Length**: 1024

### ESM-2 Configuration

| Config | Value |
|--------|-------|
| Model ID | facebook/esm2_t6_8M_UR50D |
| Embedding Dimension | 480 |
| Transformer Layers | 6 |
| Attention Heads | 8 |
| Parameters | ~8M |

> Note: ESM-2 model parameters are automatically downloaded from HuggingFace on first use. No manual parameter file configuration required.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16GB | 32GB |
| GPU | Optional | NVIDIA GPU (CUDA) |

## Dependencies

```
torch >= 2.0
transformers
sklearn
pandas
numpy
xgboost
lightgbm
matplotlib
```

## Evaluation Metrics

- **Accuracy** - Classification accuracy
- **F1 (micro)** - Micro-averaged F1 score
- **F1 (macro)** - Macro-averaged F1 score
- **Precision** - Precision score
- **Recall** - Recall score

## Performance Plots

Performance comparison plots are saved in `experiments/`:

- `loss_comparison.png` - Training and validation loss comparison
- `metrics_comparison.png` - Model performance metrics comparison
- `ec_per_class_comparison.png` - Per-class F1 scores comparison
- `accuracy_curve.png` - Accuracy curves during training

## Usage Example

```python
import torch
import numpy as np

# Load multi-task model (BNN or MLP)
model = torch.load("models/bnn_multitask.pt")

# Load features
X = np.load("data/processed/esm2_balanced/test_features.npy")

# Prediction
# ... (see scripts/predict.py for details)
```
