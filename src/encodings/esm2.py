"""ESM2 预训练语言模型编码 - 蛋白质表示学习的state-of-the-art

原理: 使用预训练的 ESM2 (Evolutionary Scale Modeling) 模型提取蛋白质序列嵌入
模型: facebook/esm2_t6_8M_UR50D
维度: 480 (隐藏层维度)

ESM2 是基于 Transformer 的蛋白质语言模型，在大量蛋白质序列上预训练，
能够学习到蛋白质的进化和结构信息。
"""
import os
import numpy as np
import torch
from typing import List, Optional
from pathlib import Path

from .base import ProteinEncoder, register_encoder

# 延迟导入 transformers
try:
    from transformers import EsmTokenizer, EsmModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@register_encoder("esm2")
class ESM2Encoder(ProteinEncoder):
    """ESM2 预训练嵌入编码器 (480 维)

    使用 facebook/esm2_t6_8M_UR50D 模型提取蛋白质序列的向量表示

    支持:
    - 本地模型缓存
    - 批量编码
    - 多种池化方式 (mean/cls/max)
    - 延迟加载 (lazy_load=True 时不立即加载模型)

    Example:
        >>> encoder = ESM2Encoder(lazy_load=True)  # 创建时不加载模型
        >>> encoder.ensure_loaded()  # 实际使用时才加载
        >>> vec = encoder.encode(seq)  # shape: (480,)
    """

    name = "esm2"
    dim = 480  # facebook/esm2_t6_8M_UR50D 隐藏层维度

    # 默认模型配置
    DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    MAX_LENGTH = 1024

    def __init__(
        self,
        model_name: str = None,
        pooling: str = "mean",
        device: str = None,
        cache_dir: Optional[str] = None,
        lazy_load: bool = False,
    ):
        """
        Args:
            model_name: 模型名称或本地路径
            pooling: 池化方式 ('mean' | 'cls' | 'max')
            device: 运行设备 ('cuda' | 'cpu')
            cache_dir: 模型缓存目录
            lazy_load: 是否延迟加载模型 (True时不在初始化时加载)
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required for ESM2Encoder. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.pooling = pooling
        self.cache_dir = cache_dir
        self.lazy_load = lazy_load

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 延迟加载标志
        self._model_loaded = False
        self.model = None
        self.tokenizer = None

        # 如果不延迟加载，立即加载
        if not lazy_load:
            self._load_model()

    def ensure_loaded(self):
        """确保模型已加载"""
        if not self._model_loaded:
            self._load_model()

    def _load_model(self):
        """加载 ESM2 模型和分词器"""
        if self._model_loaded:
            return

        print(f"[ESM2Encoder] 加载模型: {self.model_name}")
        print(f"[ESM2Encoder] 设备: {self.device}")

        # 检查本地路径
        model_path = self.model_name
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            # 尝试作为相对路径
            potential_path = Path(__file__).parent.parent.parent.parent / model_path
            if potential_path.exists():
                model_path = str(potential_path)

        try:
            self.tokenizer = EsmTokenizer.from_pretrained(model_path)
            self.model = EsmModel.from_pretrained(model_path)
        except Exception as e:
            print(f"[ESM2Encoder] 本地加载失败，尝试从 HuggingFace 下载...")
            try:
                self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
                self.model = EsmModel.from_pretrained(self.model_name)
            except Exception as e2:
                print(f"[ESM2Encoder] 错误: 无法加载模型 {e2}")
                raise

        self.model.to(self.device)
        self.model.eval()
        self._model_loaded = True
        print(f"[ESM2Encoder] 模型加载完成")

    def encode(self, sequence: str) -> np.ndarray:
        """对单条序列进行 ESM2 编码"""
        self.ensure_loaded()

        seq = self.validate_sequence(sequence)

        with torch.no_grad():
            inputs = self.tokenizer(
                seq,
                return_tensors='pt',
                max_length=self.MAX_LENGTH,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # (1, seq_len, 480)

            # 池化
            embedding = self._pool(last_hidden, inputs.get('attention_mask'))

        return embedding.cpu().numpy().squeeze()

    def encode_batch(self, sequences: List[str]) -> np.ndarray:
        """对批量序列进行 ESM2 编码"""
        if len(sequences) == 0:
            return np.array([])

        self.ensure_loaded()

        embeddings = []

        with torch.no_grad():
            for seq in sequences:
                seq = self.validate_sequence(seq)
                inputs = self.tokenizer(
                    seq,
                    return_tensors='pt',
                    max_length=self.MAX_LENGTH,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state

                embedding = self._pool(last_hidden, inputs.get('attention_mask'))
                embeddings.append(embedding.cpu().numpy().squeeze())

        return np.array(embeddings)

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """池化操作

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)

        Returns:
            (batch, hidden_dim) 池化后的向量
        """
        if self.pooling == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden = hidden_states * mask
                embedding = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                embedding = hidden_states.mean(dim=1)
        elif self.pooling == 'cls':
            embedding = hidden_states[:, 0, :]
        elif self.pooling == 'max':
            embedding = hidden_states.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return embedding

    def get_dim(self) -> int:
        return self.dim

    def info(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "description": f"ESM2 预训练嵌入 ({self.model_name})",
            "pooling": self.pooling,
            "model_name": self.model_name,
            "loaded": self._model_loaded,
        }

    def __del__(self):
        """清理模型释放显存"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
