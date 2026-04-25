"""One-Hot 氨基酸编码 - 最简单的蛋白质编码方式

原理: 统计20种标准氨基酸在序列中的出现频率
维度: 20 (每维对应一种氨基酸)
"""
import numpy as np
from typing import List

from .base import ProteinEncoder, register_encoder


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


@register_encoder("onehot")
class OneHotEncoder(ProteinEncoder):
    """氨基酸单热编码

    将20种标准氨基酸转换为20维向量，每维表示该氨基酸在序列中的出现频率。
    即 Amino Acid Composition (AAC) 编码。

    公式: feature[i] = count(aa[i]) / sequence_length

    Example:
        >>> encoder = OneHotEncoder()
        >>> seq = "MVLSPADKTN"
        >>> vec = encoder.encode(seq)  # shape: (20,)
    """

    name = "onehot"
    dim = 20

    def __init__(self):
        self.amino_acids = AMINO_ACIDS
        self.aa_to_idx = AA_TO_IDX

    def encode(self, sequence: str) -> np.ndarray:
        """对单条序列进行One-Hot编码"""
        seq = self.validate_sequence(sequence)
        length = len(seq)

        if length == 0:
            return np.zeros(self.dim)

        # 统计每种氨基酸的出现频率
        counts = np.zeros(self.dim)
        for aa in seq:
            if aa in self.aa_to_idx:
                counts[self.aa_to_idx[aa]] += 1

        # 归一化为频率
        return counts / length

    def get_dim(self) -> int:
        return self.dim

    def info(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "description": "氨基酸单热编码 - 20维频率向量",
            "amino_acids": self.amino_acids,
        }
