"""分类算法模块 - 插件式架构，支持轻松扩展新算法

默认注册:
- rf: Random Forest (随机森林)
- xgb: XGBoost (梯度提升树)
- svm: SVM (支持向量机)
- lr: Logistic Regression (逻辑回归)
- mlp: MLP (多层感知机)
- bnn: BNN (贝叶斯神经网络)

添加新算法:
1. 在本目录创建新文件, 如 my_algorithm.py
2. 继承 ProteinClassifier 基类
3. 使用 @register_classifier("my_algorithm") 装饰器
4. 自动被本模块加载
"""
from .base import (
    ProteinClassifier,
    ClassifierInfo,
    UncertainClassifier,
    ClassifierRegistry,
    register_classifier,
)

# 自动注册所有内置分类器
ClassifierRegistry.load_builtin_classifiers()

# 方便直接导入
__all__ = [
    "ProteinClassifier",
    "ClassifierInfo",
    "UncertainClassifier",
    "ClassifierRegistry",
    "register_classifier",
    "RandomForestClassifier",
    "XGBoostClassifier",
    "SVMClassifier",
    "LogisticRegressionClassifier",
    "MLPClassifier",
    "BNNClassifier",
]

# 为了向后兼容，也导出具体类
from .rf import RandomForestClassifier
from .xgb import XGBoostClassifier
from .svm import SVMClassifier
from .lr import LogisticRegressionClassifier
from .mlp import MLPClassifier
from .bnn import BNNClassifier
