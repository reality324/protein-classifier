# ProteinClassifier

Multi-task Protein Classifier - Predicting EC Numbers, Cellular Localization, and Function

## Project Structure

```
ProteinClassifier/
├── configs/
│   └── config.py                 # Configuration settings
│
├── data/
│   ├── datasets/                 # Raw datasets (parquet format)
│   └── processed/
│       └── esm2_aligned/         # ESM2 aligned features
│
├── results/
│   └── test/                     # Test results
│       ├── algorithm_comparison.png
│       ├── comparison_results.csv
│       ├── comparison_summary.png
│       ├── heatmap.png
│       └── models/               # Trained models
│           └── onehot_xgb/
│               └── xgb_best.pt
│
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── predict.py                # Prediction entry point
│   ├── compare.py                # Algorithm comparison
│   ├── train_multitask.py        # Multi-task training
│   └── generate_esm2_features.py # ESM2 feature extraction
│
├── src/
│   ├── algorithms/               # ML algorithms
│   │   ├── base.py              # Base class
│   │   ├── mlp.py               # Multi-layer Perceptron
│   │   ├── bnn.py               # Bayesian Neural Network
│   │   ├── xgb.py               # XGBoost
│   │   ├── rf.py                # Random Forest
│   │   ├── svm.py               # Support Vector Machine
│   │   └── lr.py                # Logistic Regression
│   │
│   ├── encodings/               # Feature encodings
│   │   ├── base.py              # Base encoder
│   │   ├── onehot.py            # One-hot encoding
│   │   ├── ctd.py               # CTD descriptor
│   │   └── esm2.py              # ESM2 embeddings
│   │
│   ├── pipeline/                # Training pipeline
│   │   ├── dataset.py           # Dataset handling
│   │   ├── trainer.py           # Model training
│   │   ├── evaluator.py         # Model evaluation
│   │   └── multitask.py        # Multi-task learning
│   │
│   └── utils/
│       ├── metrics.py           # Evaluation metrics
│       └── visualization.py    # Plotting utilities
│
└── README.md
```

## Features

### Supported Encodings
- **One-hot Encoding**: Traditional amino acid encoding
- **CTD Descriptor**: Composition, Transition, Distribution features
- **ESM2 Embeddings**: Protein language model embeddings from ESM-2

### Supported Algorithms
- **MLP**: Multi-layer Perceptron (PyTorch)
- **BNN**: Bayesian Neural Network for uncertainty estimation
- **XGBoost**: Gradient boosting classifier
- **Random Forest**: Ensemble learning
- **SVM**: Support Vector Machine
- **Logistic Regression**: Linear classifier

### Multi-task Learning
Simultaneously predict:
- EC Number (enzyme classification)
- Cellular Localization
- Molecular Function

## Quick Start

### 1. Feature Generation

```bash
# Generate ESM2 features
python scripts/generate_esm2_features.py

# Or use one-hot encoding
```

### 2. Training

```bash
# Single model training
python scripts/train.py --encoding onehot --algorithm xgb

# Multi-task training
python scripts/train_multitask.py

# Compare all algorithms
python scripts/compare.py
```

### 3. Prediction

```bash
python scripts/predict.py --model results/test/models/onehot_xgb/xgb_best.pt
```

## Configuration

Edit `configs/config.py` to customize:

```python
# Data settings
DATA_DIR = "data"
ENCODING = "onehot"  # or "esm2", "ctd"

# Model settings
ALGORITHM = "xgb"
HIDDEN_DIMS = [256, 128]

# Training settings
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

## Data Format

### Input Data
- CSV or Parquet files with protein sequences
- Required columns: `sequence`, `ec_number`, `localization`, `function`

### Output Results
- Trained models saved in `results/test/models/`
- Comparison plots saved in `results/test/`
- CSV summary of results

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
seaborn
```

## Performance

See `results/test/comparison_results.csv` for detailed metrics including:
- Accuracy
- F1 Score (micro/macro)
- Precision and Recall
- Per-class performance

## License

MIT License
