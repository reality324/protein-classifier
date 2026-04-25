"""XGBoost 分类器 - 高效的梯度提升树实现"""
import numpy as np
from typing import Optional, Dict, Any

from .base import ProteinClassifier, ClassifierInfo, register_classifier


@register_classifier("xgb")
class XGBoostClassifier(ProteinClassifier):
    """XGBoost 分类器

    原理: 梯度提升决策树的高效实现，通过迭代训练决策树来最小化损失函数
    优点: 精度高、处理类别不平衡、自动特征选择、速度快

    Example:
        >>> clf = XGBoostClassifier(n_estimators=200, max_depth=6)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    name = "xgb"
    info = ClassifierInfo(
        name="xgb",
        type="sklearn",
        description="XGBoost - 梯度提升树，高效的集成学习方法",
    )

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        use_xgb: bool = True,
    ):
        """
        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            subsample: 子采样比例
            colsample_bytree: 列采样比例
            min_child_weight: 最小叶子节点权重
            gamma: 最小损失减少量
            reg_alpha: L1正则化
            reg_lambda: L2正则化
            random_state: 随机种子
            n_jobs: 并行任务数
            use_xgb: 是否优先使用 xgboost 库 (False 则用 sklearn 的 GradientBoosting)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_xgb = use_xgb

        self.model = None
        self.n_classes_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "XGBoostClassifier":
        """训练 XGBoost"""
        self.n_classes_ = len(np.unique(y))

        try:
            if self.use_xgb:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    min_child_weight=self.min_child_weight,
                    gamma=self.gamma,
                    reg_alpha=self.reg_alpha,
                    reg_lambda=self.reg_lambda,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    verbosity=0,
                )
                self.model.fit(X, y, verbose=False)
            else:
                raise ImportError("Force fallback")
        except (ImportError, Exception):
            # 降级到 sklearn 的 GradientBoostingClassifier
            print("[XGBoostClassifier] xgboost 不可用，使用 sklearn GradientBoostingClassifier")
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
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
                "learning_rate": self.learning_rate,
            },
            "n_classes": self.n_classes_,
        }

    def save(self, path: str):
        """保存模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
