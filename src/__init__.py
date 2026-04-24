"""ProteinClassifier - 蛋白质分类器工具包

一个模块化的蛋白质分类框架，支持:
- 多种编码方式: One-Hot, CTD, ESM2
- 多种分类算法: RF, XGBoost, SVM, LR, MLP, BNN
- 灵活的数据处理和评估流程

快速开始:
    from src.encodings import EncoderRegistry
    from src.algorithms import ClassifierRegistry
    from src.pipeline import ProteinDataset, Trainer, Evaluator

    # 获取编码器
    encoder = EncoderRegistry.get("ctd")

    # 获取分类器
    clf = ClassifierRegistry.get("rf")

    # 训练和评估
    dataset = ProteinDataset()
    dataset.load_from_files("data/processed/esm2_balanced")

    trainer = Trainer(clf)
    trainer.train(*dataset.get_train(), *dataset.get_val())
"""
__version__ = "1.0.0"

# 子模块
from . import encodings
from . import algorithms
from . import pipeline
from . import utils

__all__ = [
    "encodings",
    "algorithms",
    "pipeline",
    "utils",
]
