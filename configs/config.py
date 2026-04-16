"""
ProteinClassifier 配置文件
"""
import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path("/home/tianwangcong/ProteinClassifier")
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# 创建目录
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============== 数据相关 ==============
# 数据目录
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"

# UniProt 数据源
UNIPROT_URLS = {
    "sprot": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz",
    "trembl": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.dat.gz",
}

# ============== 特征相关 ==============
# 嵌入方法选择
EMBEDDING_METHODS = {
    "onehot": 1,
    "esm2_8M": 2,      # ESM2 8M 参数
    "esm2_35M": 3,     # ESM2 35M 参数
    "esm2_150M": 4,    # ESM2 150M 参数
    "protbert": 5,     # ProtBERT
}

DEFAULT_EMBEDDING = "esm2_8M"  # 默认使用 ESM2 8M

# ============== 模型相关 ==============
MODEL_CONFIGS = {
    "onehot": {
        "embedding_dim": 20,
        "max_length": 10000,
    },
    "esm2_8M": {
        "model_name": "/home/tianwangcong/ProteinClassifier/models/esm2_t12_35M_UR50D",
        "embedding_dim": 480,
        "max_length": 1024,
    },
    "esm2_35M": {
        "model_name": "/home/tianwangcong/ProteinClassifier/models/esm2_t12_35M_UR50D",
        "embedding_dim": 480,
        "max_length": 1024,
    },
    "esm2_150M": {
        "model_name": "facebook/esm2_t30_150M_UR50D",
        "embedding_dim": 640,
        "max_length": 1024,
    },
    "protbert": {
        "model_name": "Rostlab/prot_bert",
        "embedding_dim": 1024,
        "max_length": 1024,
    },
}

# 分类器配置
CLASSIFIER_CONFIGS = {
    "task1_ec": {
        "name": "EC Number Prediction",
        "type": "multi_label",  # 多标签分类
        "num_classes": None,    # 动态确定
        "hidden_dims": [512, 256],
        "dropout": 0.3,
    },
    "task2_function": {
        "name": "Protein Function Prediction",
        "type": "multi_class",  # 多分类
        "num_classes": None,
        "hidden_dims": [256, 128],
        "dropout": 0.3,
    },
    "task3_localization": {
        "name": "Subcellular Localization Prediction",
        "type": "multi_class",
        "num_classes": None,
        "hidden_dims": [256, 128],
        "dropout": 0.3,
    },
}

# ============== 训练相关 ==============
TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 10,           # 早停耐心值
    "validation_split": 0.2,
    "test_split": 0.1,
    "random_seed": 42,
}

# ============== 评估指标 ==============
METRICS = {
    "binary": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "multi_class": ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    "multi_label": ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"],
}

# ============== 路径配置 ==============
FILE_PATHS = {
    "train_data": DATASETS_DIR / "train.parquet",
    "val_data": DATASETS_DIR / "val.parquet",
    "test_data": DATASETS_DIR / "test.parquet",
    "feature_cache": PROCESSED_DATA_DIR / "features",
    "ec_label_dict": DATASETS_DIR / "ec_label_dict.npy",
    "function_label_dict": DATASETS_DIR / "function_label_dict.npy",
    "localization_label_dict": DATASETS_DIR / "localization_label_dict.npy",
}

# 模型保存路径
MODEL_PATHS = {
    "isenzyme": MODELS_DIR / "isenzyme_model.h5",
    "howmany": MODELS_DIR / "howmany_model.h5",
    "ec": MODELS_DIR / "ec_model.h5",
    "function": MODELS_DIR / "function_model.h5",
    "localization": MODELS_DIR / "localization_model.h5",
    "multi_task": MODELS_DIR / "multi_task_model.h5",
}
