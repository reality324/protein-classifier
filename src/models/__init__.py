"""
模型定义 __init__.py
"""
from .multi_task_model import (
    MultiTaskProteinClassifier,
    SharedFeatureExtractor,
    TaskSpecificHead,
    MultiTaskLoss,
    HierarchicalECClassifier,
    create_model,
)
from .algorithm_comparison import (
    AlgorithmFactory,
    AlgorithmComparator,
    AlgorithmConfig,
    RandomForestAlgorithm,
    XGBoostAlgorithm,
    SVMAlgorithm,
    LogisticRegressionAlgorithm,
    NeuralNetworkAlgorithm,
)

__all__ = [
    # Multi-task model
    'MultiTaskProteinClassifier',
    'SharedFeatureExtractor',
    'TaskSpecificHead',
    'MultiTaskLoss',
    'HierarchicalECClassifier',
    'create_model',
    # Algorithm comparison
    'AlgorithmFactory',
    'AlgorithmComparator',
    'AlgorithmConfig',
    'RandomForestAlgorithm',
    'XGBoostAlgorithm',
    'SVMAlgorithm',
    'LogisticRegressionAlgorithm',
    'NeuralNetworkAlgorithm',
]
