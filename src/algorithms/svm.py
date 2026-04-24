"""SVM 分类器 - 支持向量机"""
import numpy as np
from typing import Optional, Dict, Any

from .base import ProteinClassifier, ClassifierInfo, register_classifier


@register_classifier("svm")
class SVMClassifier(ProteinClassifier):
    """SVM 分类器

    原理: 寻找最大间隔分类超平面，将不同类别的样本分开
    优点: 在高维空间效果好、通过核函数处理非线性问题
    缺点: 对大规模数据训练慢、对参数敏感

    Example:
        >>> clf = SVMClassifier(kernel='rbf')
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    name = "svm"
    info = ClassifierInfo(
        name="svm",
        type="sklearn",
        description="SVM - 支持向量机，核函数支持非线性分类",
    )

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        probability: bool = True,
        random_state: int = 42,
    ):
        """
        Args:
            kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
            C: 正则化参数
            gamma: 核系数 ('scale', 'auto' 或 float)
            degree: 多项式核的度数
            coef0: 核函数常数项
            probability: 是否启用概率估计
            random_state: 随机种子
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.probability = probability
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
    ) -> "SVMClassifier":
        """训练 SVM (包含标准化)"""
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        self.n_classes_ = len(np.unique(y))

        # SVM 对特征尺度敏感，需要标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            probability=self.probability,
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
                "kernel": self.kernel,
                "C": self.C,
                "gamma": self.gamma,
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
