"""编码方式基类和注册表 - 实现插件式编码扩展

所有编码方式都继承 ProteinEncoder 基类，并通过 register_encoder 装饰器注册。
添加新编码只需创建新文件并装饰，无需修改本文件。
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import hashlib


class ProteinEncoder(ABC):
    """蛋白质编码器基类

    所有编码方式必须实现:
    - encode: 对单条序列编码
    - encode_batch: 对批量序列编码
    - get_dim: 返回编码维度
    """

    name: str = "base"  # 编码名称，需在子类覆盖
    dim: int = 0        # 编码维度，需在子类覆盖

    @abstractmethod
    def encode(self, sequence: str) -> np.ndarray:
        """对单条蛋白质序列进行编码

        Args:
            sequence: 蛋白质氨基酸序列 (如 "MVLSPADKTNVKAAWGKG...")

        Returns:
            np.ndarray: 编码后的特征向量，shape = (dim,)
        """
        pass

    def encode_batch(self, sequences: List[str]) -> np.ndarray:
        """对批量蛋白质序列进行编码

        默认实现为循环调用 encode，可被子类优化

        Args:
            sequences: 蛋白质序列列表

        Returns:
            np.ndarray: 编码后的特征矩阵，shape = (n_samples, dim)
        """
        return np.array([self.encode(seq) for seq in sequences])

    @abstractmethod
    def get_dim(self) -> int:
        """返回编码特征维度"""
        pass

    def validate_sequence(self, sequence: str) -> str:
        """验证并标准化序列

        Args:
            sequence: 原始序列

        Returns:
            标准化后的序列 (大写、移除空格/换行、移除非标准氨基酸)
        """
        seq = sequence.upper().replace(" ", "").replace("\n", "").replace("\r", "")
        # 移除非标准氨基酸（保留20种标准氨基酸）
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        seq = "".join(aa for aa in seq if aa in valid_aas)
        if len(seq) == 0:
            raise ValueError("Sequence contains no valid amino acids after filtering")
        return seq

    def get_cache_key(self, sequence: str) -> str:
        """生成序列编码缓存key"""
        return hashlib.md5(f"{self.name}:{sequence}".encode()).hexdigest()

    def info(self) -> Dict[str, Any]:
        """返回编码器信息"""
        return {
            "name": self.name,
            "dim": self.dim,
            "type": self.__class__.__name__,
        }


# ============== 注册表 ==============

_ENCODING_REGISTRY: Dict[str, type] = {}


def register_encoder(name: str):
    """编码器注册装饰器

    使用方式:
        @register_encoder("my_encoder")
        class MyEncoder(ProteinEncoder):
            ...

    注册后可通过 EncoderRegistry.get("my_encoder") 获取实例
    """
    def decorator(cls):
        if name in _ENCODING_REGISTRY:
            raise ValueError(f"Encoder '{name}' already registered")
        _ENCODING_REGISTRY[name] = cls
        return cls
    return decorator


class EncoderRegistry:
    """编码器注册表管理器"""

    _instance = None

    @classmethod
    def get(cls, name: str) -> ProteinEncoder:
        """获取指定名称的编码器实例"""
        if name not in _ENCODING_REGISTRY:
            available = list(_ENCODING_REGISTRY.keys())
            raise ValueError(f"Encoder '{name}' not found. Available: {available}")
        return _ENCODING_REGISTRY[name]()

    @classmethod
    def list_encodings(cls) -> List[str]:
        """列出所有已注册的编码方式"""
        return list(_ENCODING_REGISTRY.keys())

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """获取编码器信息"""
        if name not in _ENCODING_REGISTRY:
            raise ValueError(f"Encoder '{name}' not found")
        encoder = _ENCODING_REGISTRY[name]()
        return encoder.info()

    @classmethod
    def get_all_info(cls) -> List[Dict[str, Any]]:
        """获取所有编码器信息 (延迟加载)"""
        info_list = []
        for name in _ENCODING_REGISTRY:
            try:
                # 对于需要网络的编码器，使用 lazy_load
                if name == "esm2":
                    encoder = _ENCODING_REGISTRY[name](lazy_load=True)
                else:
                    encoder = _ENCODING_REGISTRY[name]()
                info_list.append(encoder.info())
            except Exception as e:
                print(f"[EncoderRegistry] Warning: Failed to get info for '{name}': {e}")
        return info_list

    @classmethod
    def register(cls, name: str, encoder_cls: type):
        """手动注册编码器"""
        if not issubclass(encoder_cls, ProteinEncoder):
            raise TypeError(f"{encoder_cls} must be a subclass of ProteinEncoder")
        _ENCODING_REGISTRY[name] = encoder_cls

    @classmethod
    def load_builtin_encodings(cls):
        """加载所有内置编码器"""
        # 延迟导入避免循环依赖
        from .onehot import OneHotEncoder
        from .ctd import CTDEncoder
        from .esm2 import ESM2Encoder

    @classmethod
    def auto_register(cls):
        """自动发现并注册项目中的编码器

        会自动导入 encodings 目录下的所有模块
        """
        import importlib
        import pkgutil
        from pathlib import Path

        pkg_dir = Path(__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(pkg_dir)]):
            if module_name in ("base", "__pycache__"):
                continue
            importlib.import_module(f".{module_name}", package=__name__)
