"""Logistic Regression 分类器 - 逻辑回归"""
import numpy as np
from typing import Optional, Dict, Any

from .base import ProteinClassifier, ClassifierInfo, register_classifier


@register_classifier("lr")
class LogisticRegressionClassifier(ProteinClassifier):
    """逻辑回归分类器

    原理: 使用 sigmoid 函数将线性组合映射到 [0,1]，进行二分类或多分类
    优点: 简单快速、可解释性强、输出概率
    缺点: 无法捕捉非线性关系

    Example:
        >>> clf = LogisticRegressionClassifier(max_iter=1000)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    name = "lr"
    info = ClassifierInfo(
        name="lr",
        type="sklearn",
        description="Logistic Regression - 逻辑回归，线性分类器",
    )

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        """
        Args:
            penalty: 正则化类型 ('l1', 'l2', 'elasticnet', 'none')
            C: 正则化强度的倒数 (值越小正则化越强)
            solver: 优化算法 ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
            max_iter: 最大迭代次数
            random_state: 随机种子
        """
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

        self.model = None
        self.scaler = None
        self.n_classes_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "LogisticRegressionClassifier":
        """训练逻辑回归 (包含标准化)"""
        from sklearn.linear_model import LogisticRegression as SklearnLR
        from sklearn.preprocessing import StandardScaler

        self.n_classes_ = len(np.unique(y))

        # 标准化有助于收敛
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # L1 + L2 正则化需要 saga solver
        if self.penalty == 'elasticnet':
            solver = 'saga'
        elif self.penalty == 'l1':
            solver = 'liblinear'
        else:
            solver = self.solver

        self.model = SklearnLR(
            penalty=self.penalty,
            C=self.C,
            solver=solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight='balanced',
        )

        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_info(self) -> Dict[str, Any]:
        return {
            **super().get_info(),
            "params": {
                "penalty": self.penalty,
                "C": self.C,
                "solver": self.solver,
            },
            "n_classes": self.n_classes_,
        }

    def save(self, path: str):
        """保存模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)

    def load(self, path: str):
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
