"""分类算法基类和注册表 - 实现插件式算法扩展

所有分类算法都继承 ProteinClassifier 基类，并通过 register_classifier 装饰器注册。
添加新算法只需创建新文件并装饰，无需修改本文件。
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ClassifierInfo:
    """分类器信息"""
    name: str
    type: str  # "sklearn" | "pytorch"
    description: str
    supports_uncertainty: bool = False
    requires_gpu: bool = False


class ProteinClassifier(ABC):
    """蛋白质分类器基类

    所有分类算法必须实现:
    - fit: 训练模型
    - predict: 预测类别
    - predict_proba: 预测概率
    - get_info: 返回分类器信息
    """

    name: str = "base"
    info: ClassifierInfo = None

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "ProteinClassifier":
        """训练模型

        Args:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,) 或 (n_samples, n_tasks) for multi-label
            X_val: 验证特征 (可选)
            y_val: 验证标签 (可选)

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            预测类别 (n_samples,) 或 (n_samples, n_tasks)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            预测概率 (n_samples, n_classes) 或 (n_samples, n_tasks, n_classes)
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """返回分类器信息"""
        if self.info is None:
            self.info = ClassifierInfo(
                name=self.name,
                type="unknown",
                description="",
            )
        return {
            "name": self.info.name,
            "type": self.info.type,
            "description": self.info.description,
            "supports_uncertainty": self.info.supports_uncertainty,
        }

    def save(self, path: str):
        """保存模型"""
        raise NotImplementedError(f"{self.name} does not support saving")

    def load(self, path: str):
        """加载模型"""
        raise NotImplementedError(f"{self.name} does not support loading")


class UncertainClassifier(ABC):
    """支持不确定性估计的分类器扩展接口

    实现此接口的分类器可以提供预测的不确定性估计
    """

    @abstractmethod
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """带不确定性估计的预测

        Args:
            X: 特征矩阵
            n_samples: Monte Carlo 采样次数

        Returns:
            (predictions, uncertainties, probs)
            - predictions: 预测类别 (n_samples,)
            - uncertainties: 不确定性分数 (n_samples,) - 熵或方差
            - probs: 平均预测概率 (n_samples, n_classes)
        """
        pass


# ============== 注册表 ==============

_CLASSIFIER_REGISTRY: Dict[str, type] = {}


def register_classifier(name: str, category: str = "default"):
    """分类器注册装饰器

    使用方式:
        @register_classifier("my_classifier")
        class MyClassifier(ProteinClassifier):
            ...

    注册后可通过 ClassifierRegistry.get("my_classifier") 获取实例
    """
    def decorator(cls):
        if name in _CLASSIFIER_REGISTRY:
            raise ValueError(f"Classifier '{name}' already registered")
        _CLASSIFIER_REGISTRY[name] = cls
        return cls
    return decorator


class ClassifierRegistry:
    """分类器注册表管理器"""

    @classmethod
    def get(cls, name: str, **kwargs) -> ProteinClassifier:
        """获取指定名称的分类器实例"""
        if name not in _CLASSIFIER_REGISTRY:
            available = list(_CLASSIFIER_REGISTRY.keys())
            raise ValueError(
                f"Classifier '{name}' not found. Available: {available}"
            )
        return _CLASSIFIER_REGISTRY[name](**kwargs)

    @classmethod
    def list_classifiers(cls) -> List[str]:
        """列出所有已注册的分类器"""
        return list(_CLASSIFIER_REGISTRY.keys())

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """获取分类器信息"""
        if name not in _CLASSIFIER_REGISTRY:
            raise ValueError(f"Classifier '{name}' not found")
        clf = _CLASSIFIER_REGISTRY[name]()
        return clf.get_info()

    @classmethod
    def get_all_info(cls) -> List[Dict[str, Any]]:
        """获取所有分类器信息"""
        info_list = []
        for name in _CLASSIFIER_REGISTRY:
            try:
                clf = _CLASSIFIER_REGISTRY[name]()
                info_list.append(clf.get_info())
            except Exception as e:
                print(f"[ClassifierRegistry] Warning: Failed to get info for '{name}': {e}")
        return info_list

    @classmethod
    def register(cls, name: str, clf_cls: type):
        """手动注册分类器"""
        if not issubclass(clf_cls, ProteinClassifier):
            raise TypeError(f"{clf_cls} must be a subclass of ProteinClassifier")
        _CLASSIFIER_REGISTRY[name] = clf_cls

    @classmethod
    def load_builtin_classifiers(cls):
        """加载所有内置分类器"""
        # 延迟导入避免循环依赖
        from .rf import RandomForestClassifier
        from .xgb import XGBoostClassifier
        from .svm import SVMClassifier
        from .lr import LogisticRegressionClassifier
        from .mlp import MLPClassifier
        from .bnn import BNNClassifier

    @classmethod
    def auto_register(cls):
        """自动发现并注册项目中的分类器"""
        import importlib
        import pkgutil
        from pathlib import Path

        pkg_dir = Path(__file__).parent
        for _, module_name, _ in pkgutil.iterModules([str(pkg_dir)]):
            if module_name in ("base", "__pycache__"):
                continue
            importlib.import_module(f".{module_name}", package=__name__)
