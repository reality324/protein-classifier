"""Random Forest 分类器 - 基于决策树的集成学习方法"""
import numpy as np
from typing import Optional, Dict, Any
import pickle

from .base import ProteinClassifier, ClassifierInfo, register_classifier


@register_classifier("rf")
class RandomForestClassifier(ProteinClassifier):
    """随机森林分类器

    原理: 通过构建多棵决策树并集成它们的预测结果来进行分类
    优点: 可解释性强、不容易过拟合、能处理高维数据

    Example:
        >>> clf = RandomForestClassifier(n_estimators=200)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    name = "rf"
    info = ClassifierInfo(
        name="rf",
        type="sklearn",
        description="Random Forest - 随机森林，基于决策树集成",
    )

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            min_samples_split: 节点分裂所需的最小样本数
            min_samples_leaf: 叶节点最小样本数
            max_features: 分裂时考虑的最大特征数
            random_state: 随机种子
            n_jobs: 并行任务数
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = None
        self.n_classes_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "RandomForestClassifier":
        """训练随机森林"""
        from sklearn.ensemble import RandomForestClassifier as SklearnRF

        self.n_classes_ = len(np.unique(y))

        self.model = SklearnRF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight='balanced',  # 处理类别不平衡
        )

        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.model.predict_proba(X)

    def get_info(self) -> Dict[str, Any]:
        return {
            **super().get_info(),
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
            },
            "n_classes": self.n_classes_,
        }

    def save(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.n_classes_ = self.model.n_classes_
