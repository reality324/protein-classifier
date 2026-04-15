"""
数据增强模块 - 蛋白质序列数据增强
"""
import random
from typing import List, Callable
import numpy as np


class ProteinAugmenter:
    """蛋白质序列数据增强器"""
    
    # 标准氨基酸
    AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
    
    # 保守替换 (根据氨基酸性质)
    CONSERVATIVE_REPLACEMENTS = {
        'A': ['V', 'S', 'G'],
        'R': ['K', 'H'],
        'N': ['D', 'Q', 'H'],
        'D': ['E', 'N'],
        'C': ['S', 'M'],
        'E': ['D', 'Q'],
        'Q': ['E', 'K'],
        'G': ['A', 'S'],
        'H': ['Q', 'N', 'R'],
        'I': ['L', 'V', 'M'],
        'L': ['I', 'V', 'M'],
        'K': ['R', 'Q'],
        'M': ['L', 'I', 'V'],
        'F': ['Y', 'W'],
        'P': ['A', 'G'],
        'S': ['T', 'A', 'N'],
        'T': ['S', 'A'],
        'W': ['F', 'Y'],
        'Y': ['F', 'W'],
        'V': ['I', 'L', 'A'],
    }
    
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
    
    def mutate(
        self,
        sequence: str,
        rate: float = 0.01,
        conservative: bool = True,
    ) -> str:
        """随机突变
        
        Args:
            sequence: 原始序列
            rate: 突变率
            conservative: 是否使用保守替换
        
        Returns:
            增强后的序列
        """
        seq_list = list(sequence.upper())
        
        for i in range(len(seq_list)):
            if random.random() < rate:
                aa = seq_list[i]
                
                if conservative and aa in self.CONSERVATIVE_REPLACEMENTS:
                    seq_list[i] = random.choice(
                        self.CONSERVATIVE_REPLACEMENTS[aa]
                    )
                else:
                    seq_list[i] = random.choice(self.AMINO_ACIDS)
        
        return ''.join(seq_list)
    
    def random_insertion(
        self,
        sequence: str,
        n: int = 1,
    ) -> str:
        """随机插入氨基酸
        
        Args:
            sequence: 原始序列
            n: 插入数量
        
        Returns:
            增强后的序列
        """
        seq_list = list(sequence.upper())
        
        for _ in range(n):
            pos = random.randint(0, len(seq_list))
            seq_list.insert(pos, random.choice(self.AMINO_ACIDS))
        
        return ''.join(seq_list)
    
    def random_deletion(
        self,
        sequence: str,
        n: int = 1,
    ) -> str:
        """随机删除氨基酸
        
        Args:
            sequence: 原始序列
            n: 删除数量
        
        Returns:
            增强后的序列
        """
        seq_list = list(sequence.upper())
        
        # 确保删除后序列长度足够
        max_delete = min(n, len(seq_list) - 10)
        
        for _ in range(max_delete):
            pos = random.randint(0, len(seq_list) - 1)
            seq_list.pop(pos)
        
        return ''.join(seq_list)
    
    def swap(
        self,
        sequence: str,
        n: int = 1,
    ) -> str:
        """随机交换相邻氨基酸
        
        Args:
            sequence: 原始序列
            n: 交换次数
        
        Returns:
            增强后的序列
        """
        seq_list = list(sequence.upper())
        
        for _ in range(n):
            if len(seq_list) < 2:
                break
            pos = random.randint(0, len(seq_list) - 2)
            seq_list[pos], seq_list[pos + 1] = seq_list[pos + 1], seq_list[pos]
        
        return ''.join(seq_list)
    
    def truncate(
        self,
        sequence: str,
        min_length: int = 50,
        max_length: int = 500,
    ) -> str:
        """序列截断
        
        Args:
            sequence: 原始序列
            min_length: 最小长度
            max_length: 最大长度
        
        Returns:
            截断后的序列
        """
        seq_len = len(sequence)
        
        if seq_len <= min_length:
            return sequence.upper()
        
        # 随机选择截断长度
        target_len = random.randint(min_length, min(seq_len, max_length))
        
        # 随机选择起始位置
        start = random.randint(0, seq_len - target_len)
        end = start + target_len
        
        return sequence[start:end].upper()
    
    def reverse(
        self,
        sequence: str,
    ) -> str:
        """序列反转"""
        return sequence[::-1].upper()
    
    def shuffle_subsequence(
        self,
        sequence: str,
        n_shuffles: int = 1,
    ) -> str:
        """打乱子序列
        
        Args:
            sequence: 原始序列
            n_shuffles: 打乱次数
        
        Returns:
            增强后的序列
        """
        seq_list = list(sequence.upper())
        
        for _ in range(n_shuffles):
            if len(seq_list) < 5:
                continue
            
            # 选择子序列
            start = random.randint(0, len(seq_list) - 5)
            end = random.randint(start + 5, len(seq_list))
            
            # 打乱
            subsequence = seq_list[start:end]
            random.shuffle(subsequence)
            seq_list[start:end] = subsequence
        
        return ''.join(seq_list)
    
    def augment(
        self,
        sequence: str,
        methods: List[str] = None,
        n_augments: int = 1,
    ) -> List[str]:
        """组合增强
        
        Args:
            sequence: 原始序列
            methods: 增强方法列表
            n_augments: 每个序列增强数量
        
        Returns:
            增强后的序列列表
        """
        if methods is None:
            methods = ['mutate', 'swap', 'truncate']
        
        augmented = [sequence.upper()]
        
        for _ in range(n_augments):
            aug_seq = sequence.upper()
            
            for method in methods:
                if method == 'mutate':
                    aug_seq = self.mutate(aug_seq, rate=0.01)
                elif method == 'swap':
                    aug_seq = self.swap(aug_seq, n=1)
                elif method == 'truncate':
                    aug_seq = self.truncate(aug_seq)
                elif method == 'shuffle':
                    aug_seq = self.shuffle_subsequence(aug_seq)
            
            augmented.append(aug_seq)
        
        return augmented
    
    def augment_batch(
        self,
        sequences: List[str],
        labels: np.ndarray = None,
        n_augments: int = 1,
        methods: List[str] = None,
    ) -> tuple:
        """批量增强
        
        Args:
            sequences: 序列列表
            labels: 标签数组
            n_augments: 每个序列增强数量
            methods: 增强方法
        
        Returns:
            (增强序列, 对应标签)
        """
        augmented_seqs = []
        augmented_labels = []
        
        for i, seq in enumerate(sequences):
            aug_seqs = self.augment(seq, methods, n_augments)
            augmented_seqs.extend(aug_seqs)
            
            if labels is not None:
                for _ in range(len(aug_seqs)):
                    augmented_labels.append(labels[i])
        
        return augmented_seqs, np.array(augmented_labels) if labels is not None else None


# 预定义的增强策略
AUGMENTATION_STRATEGIES = {
    'light': ['mutate'],
    'medium': ['mutate', 'swap'],
    'heavy': ['mutate', 'swap', 'truncate', 'shuffle'],
    'aggressive': ['mutate', 'swap', 'truncate'],
}


if __name__ == "__main__":
    # 测试
    augmenter = ProteinAugmenter(seed=42)
    
    seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
    
    print("原始序列:", seq)
    print("长度:", len(seq))
    
    print("\n突变:", augmenter.mutate(seq, rate=0.05))
    print("交换:", augmenter.swap(seq, n=2))
    print("截断:", augmenter.truncate(seq, min_length=30))
    
    print("\n组合增强:")
    augmented = augmenter.augment(seq, methods=['mutate', 'swap'], n_augments=3)
    for i, a in enumerate(augmented):
        print(f"  {i}: {a[:50]}... ({len(a)} aa)")
