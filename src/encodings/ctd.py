"""CTD 编码 (Composition-Transition-Distribution) - 蛋白质物化性质编码

原理: 基于氨基酸的物理化学性质进行编码，总维度 147 维:
- C (Composition): 21维 (20种氨基酸的组成比例 + gap)
- T (Transition): 21维 (7组 × 3种转换类型)
- D (Distribution): 105维 (7组 × 5个位置点 × 3个统计量)

转换类型: 从该组出去、从外面进来、在组内相邻
位置点: 0%, 25%, 50%, 75%, 100% 的累积比例
统计量: 头部位置、中间位置、尾部位置

7个属性组:
- hydrophobicity (疏水性): AILMFWV
- hydrophilicity (亲水性): RNEDQGHKSTY
- charge_pos (正电荷): KR
- charge_neg (负电荷): DE
- polar (极性不带电): NQST
- proline (脯氨酸): P
- turn_inducing (促旋转型): GNP
"""
import numpy as np
from typing import List
from collections import Counter

from .base import ProteinEncoder, register_encoder


@register_encoder("ctd")
class CTDEncoder(ProteinEncoder):
    """CTD 编码器 (147 维)

    将蛋白质序列编码为 Composition, Transition, Distribution 特征
    基于 iFeature/ProsAIC 的标准实现

    Example:
        >>> encoder = CTDEncoder()
        >>> seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        >>> vec = encoder.encode(seq)  # shape: (147,)
    """

    name = "ctd"
    dim = 147

    # 20种标准氨基酸
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

    # 标准 CTD 分组 (7组，每组对应不同的物理化学性质)
    CTD_GROUPS = {
        'hydrophobicity': 'AILMFWV',           # 疏水性
        'hydrophilicity': 'RNEDQGHKSTY',       # 亲水性
        'charge_pos': 'KR',                    # 正电荷
        'charge_neg': 'DE',                    # 负电荷
        'polar': 'NQST',                       # 极性不带电
        'proline': 'P',                        # 脯氨酸
        'turn_inducing': 'GNP',                # 促旋转型
    }

    def __init__(self):
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.AMINO_ACIDS)}
        self._build_group_mapping()

    def _build_group_mapping(self):
        """构建氨基酸到属性组的映射"""
        self.aa_to_group = {}
        for group_name, members in self.CTD_GROUPS.items():
            for aa in members:
                self.aa_to_group[aa] = group_name

    def encode(self, sequence: str) -> np.ndarray:
        """对单条序列进行CTD编码"""
        seq = self.validate_sequence(sequence)
        seq_list = list(seq)

        if len(seq_list) == 0:
            return np.zeros(self.dim)

        features = []

        # 1. Composition (C) - 21维
        features.extend(self._compute_composition(seq_list))

        # 2. Transition (T) - 21维
        features.extend(self._compute_transition(seq_list))

        # 3. Distribution (D) - 105维
        features.extend(self._compute_distribution(seq_list))

        return np.array(features)

    def get_dim(self) -> int:
        return self.dim

    def _compute_composition(self, seq: List[str]) -> np.ndarray:
        """计算氨基酸组成 (21维)

        20种标准氨基酸的频率 + gap (非标准氨基酸) 的比例
        """
        n = len(seq)
        comp = np.zeros(21)  # 20 AA + gap

        for aa in seq:
            if aa in self.aa_to_idx:
                comp[self.aa_to_idx[aa]] += 1

        if n > 0:
            comp[:20] /= n

        comp[20] = 0.0  # gap = 0 (已过滤掉非标准氨基酸)
        return comp

    def _compute_transition(self, seq: List[str]) -> np.ndarray:
        """计算氨基酸转换 (21维)

        对于7个属性组，每组计算3种转换频率:
        - 1->0: 从该组转到非该组
        - 0->1: 从非该组转到该组
        - 1->1: 在该组内相邻
        """
        n = len(seq)
        if n < 2:
            return np.zeros(21)

        # 属性序列
        props = []
        for aa in seq:
            props.append(self.aa_to_group.get(aa, None))

        trans = []
        group_names = list(self.CTD_GROUPS.keys())

        for group_name in group_names:
            n_go_out = 0
            n_go_in = 0
            n_within = 0
            total_pairs = 0

            for i in range(n - 1):
                p1, p2 = props[i], props[i + 1]

                if p1 is None or p2 is None:
                    continue

                if p1 == group_name and p2 == group_name:
                    n_within += 1
                    total_pairs += 1
                elif p1 == group_name and p2 != group_name:
                    n_go_out += 1
                    total_pairs += 1
                elif p1 != group_name and p2 == group_name:
                    n_go_in += 1
                    total_pairs += 1
                else:
                    total_pairs += 1

            if total_pairs > 0:
                trans.extend([
                    n_go_out / total_pairs,
                    n_go_in / total_pairs,
                    n_within / total_pairs
                ])
            else:
                trans.extend([0.0, 0.0, 0.0])

        return np.array(trans)

    def _compute_distribution(self, seq: List[str]) -> np.ndarray:
        """计算氨基酸位置分布 (105维)

        对于7个属性组，计算5个累积百分比位置的头、中、尾:
        - 每个组有5个百分位点: 0%, 25%, 50%, 75%, 100%
        - 每个百分位点有3个统计量: 头部位置、中间位置、尾部位置
        """
        n = len(seq)
        if n == 0:
            return np.zeros(105)

        dist = []
        group_names = list(self.CTD_GROUPS.keys())

        for group_name in group_names:
            group_aa = set(self.CTD_GROUPS[group_name])

            # 找到该组氨基酸的位置
            positions = [i for i, aa in enumerate(seq) if aa in group_aa]
            n_aa = len(positions)

            if n_aa == 0:
                dist.extend([0.0] * 15)
                continue

            # 对5个百分位点计算
            for target_pct in [0, 25, 50, 75, 100]:
                target_idx = target_pct / 100 * (n_aa - 1)

                lower_idx = int(target_idx)
                upper_idx = min(lower_idx + 1, n_aa - 1)
                frac = target_idx - lower_idx

                # 计算相对位置
                rel_pos = ((positions[lower_idx] * (1 - frac) +
                           positions[upper_idx] * frac) + 1) / n

                # 计算头部/中间/尾部分数
                head_score = 1.0 if rel_pos < 1/3 else max(0, 1 - (rel_pos - 1/3) * 1.5)
                mid_score = 1.0 if 1/3 <= rel_pos <= 2/3 else max(0, 1 - abs(rel_pos - 0.5) * 3)
                tail_score = 1.0 if rel_pos > 2/3 else max(0, (rel_pos - 1/3) * 1.5)

                dist.extend([head_score, mid_score, tail_score])

        return np.array(dist[:105])

    def info(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "description": "CTD编码 - 组成/转变/分布 (147维)",
            "groups": list(self.CTD_GROUPS.keys()),
        }
