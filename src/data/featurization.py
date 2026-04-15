"""
蛋白质特征提取模块
支持多种嵌入方法: One-Hot, ESM2, ProtBERT
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import (
    AutoTokenizer, AutoModel, 
    EsmTokenizer, EsmModel
)
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import (
    PROCESSED_DATA_DIR, MODEL_CONFIGS, DEFAULT_EMBEDDING
)


class ProteinFeatureExtractor:
    """蛋白质序列特征提取基类"""
    
    def __init__(self, model_name: str, max_length: int = 1024):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract(self, sequences: List[str]) -> np.ndarray:
        """提取特征"""
        raise NotImplementedError
    
    def _batch_encode(self, sequences: List[str], batch_size: int = 32):
        """批量编码"""
        all_embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            embeddings = self.extract(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)


class OneHotFeatureExtractor(ProteinFeatureExtractor):
    """One-Hot 编码特征提取器"""
    
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    def __init__(self):
        super().__init__('onehot')
        # 构建 amino acid to index 映射
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.AMINO_ACIDS)}
        self.n_tokens = len(self.AMINO_ACIDS)
    
    def extract(self, sequences: List[str]) -> np.ndarray:
        """提取 One-Hot 特征
        
        策略: 对序列中的氨基酸进行 One-Hot 编码，然后求和并归一化
        """
        features = []
        
        for seq in sequences:
            seq_upper = seq.upper()
            seq_len = len(seq_upper)
            
            # One-Hot 编码
            one_hot = np.zeros((seq_len, self.n_tokens))
            for i, aa in enumerate(seq_upper):
                if aa in self.aa_to_idx:
                    one_hot[i, self.aa_to_idx[aa]] = 1
            
            # 求和并归一化
            seq_feature = one_hot.sum(axis=0) / max(seq_len, 1)
            features.append(seq_feature)
        
        return np.array(features)
    
    def extract_kmer(self, sequences: List[str], k: int = 3) -> np.ndarray:
        """提取 k-mer 特征
        
        Args:
            sequences: 序列列表
            k: k-mer 大小
        
        Returns:
            k-mer 频率特征
        """
        from itertools import product
        
        # 生成所有 k-mer
        kmers = [''.join(p) for p in product(self.AMINO_ACIDS, repeat=k)]
        kmer_to_idx = {kmer: idx for idx, kmer in enumerate(kmers)}
        n_kmers = len(kmers)
        
        features = []
        for seq in sequences:
            seq_upper = seq.upper()
            seq_len = len(seq_upper)
            
            # 统计 k-mer 频率
            kmer_counts = np.zeros(n_kmers)
            
            for i in range(seq_len - k + 1):
                kmer = seq_upper[i:i+k]
                if kmer in kmer_to_idx:
                    kmer_counts[kmer_to_idx[kmer]] += 1
            
            # 归一化
            if seq_len > 0:
                kmer_counts = kmer_counts / (seq_len - k + 1)
            
            features.append(kmer_counts)
        
        return np.array(features)


class ESMFeatureExtractor(ProteinFeatureExtractor):
    """ESM2 嵌入特征提取器"""
    
    def __init__(self, embedding_type: str = "esm2_8M", pooling: str = "mean"):
        """
        Args:
            embedding_type: ESM 模型类型
            pooling: 池化方式 ['mean', 'cls', 'max']
        """
        config = MODEL_CONFIGS.get(embedding_type, MODEL_CONFIGS['esm2_8M'])
        super().__init__(config['model_name'], config['max_length'])
        self.embedding_dim = config['embedding_dim']
        self.pooling = pooling
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        print(f"加载 ESM 模型: {self.model_name}")
        self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
        self.model = EsmModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"模型已加载，设备: {self.device}")
    
    def extract(self, sequences: List[str]) -> np.ndarray:
        """提取 ESM2 嵌入
        
        Args:
            sequences: 蛋白质序列列表
        
        Returns:
            (n_samples, embedding_dim) 嵌入矩阵
        """
        all_embeddings = []
        
        with torch.no_grad():
            for seq in tqdm(sequences, desc="提取 ESM2 嵌入"):
                # 编码
                inputs = self.tokenizer(
                    seq,
                    return_tensors='pt',
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
                
                # 池化
                if self.pooling == 'mean':
                    # Mean pooling (考虑 attention mask)
                    attention_mask = inputs['attention_mask']
                    hidden = last_hidden * attention_mask.unsqueeze(-1)
                    embedding = hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                elif self.pooling == 'cls':
                    # CLS token
                    embedding = last_hidden[:, 0, :]
                elif self.pooling == 'max':
                    # Max pooling
                    embedding = last_hidden.max(dim=1).values
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")
                
                all_embeddings.append(embedding.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def extract_per_residue(self, sequences: List[str]) -> np.ndarray:
        """提取每个残基的嵌入 (用于更细粒度的分析)"""
        all_embeddings = []
        
        with torch.no_grad():
            for seq in tqdm(sequences, desc="提取残基级嵌入"):
                inputs = self.tokenizer(
                    seq,
                    return_tensors='pt',
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                all_embeddings.append(outputs.last_hidden_state.cpu().numpy())
        
        return all_embeddings


class ProtBERTFeatureExtractor(ProteinFeatureExtractor):
    """ProtBERT 嵌入特征提取器"""
    
    def __init__(self, pooling: str = "mean"):
        config = MODEL_CONFIGS['protbert']
        super().__init__(config['model_name'], config['max_length'])
        self.embedding_dim = config['embedding_dim']
        self.pooling = pooling
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        print(f"加载 ProtBERT 模型: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"模型已加载，设备: {self.device}")
    
    def extract(self, sequences: List[str]) -> np.ndarray:
        """提取 ProtBERT 嵌入"""
        # 格式化为 FASTA 格式 (ProtBERT 要求)
        formatted = [f">seq\n{seq}" for seq in sequences]
        
        all_embeddings = []
        
        with torch.no_grad():
            for seq in tqdm(formatted, desc="提取 ProtBERT 嵌入"):
                # 编码
                inputs = self.tokenizer(
                    seq,
                    return_tensors='pt',
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                
                # 池化
                if self.pooling == 'mean':
                    attention_mask = inputs['attention_mask']
                    hidden = last_hidden * attention_mask.unsqueeze(-1)
                    embedding = hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                elif self.pooling == 'cls':
                    embedding = last_hidden[:, 0, :]
                else:
                    embedding = last_hidden.mean(dim=1)
                
                all_embeddings.append(embedding.cpu().numpy())
        
        return np.vstack(all_embeddings)


def get_feature_extractor(
    method: str = DEFAULT_EMBEDDING, 
    pooling: str = "mean"
) -> ProteinFeatureExtractor:
    """获取特征提取器
    
    Args:
        method: 嵌入方法 ['onehot', 'esm2_8M', 'esm2_35M', 'esm2_150M', 'protbert']
        pooling: 池化方式
    
    Returns:
        特征提取器实例
    """
    if method == 'onehot':
        return OneHotFeatureExtractor()
    elif method.startswith('esm2'):
        return ESMFeatureExtractor(method, pooling)
    elif method == 'protbert':
        return ProtBERTFeatureExtractor(pooling)
    else:
        raise ValueError(f"Unknown method: {method}")


def batch_extract_features(
    sequences: List[str],
    method: str = DEFAULT_EMBEDDING,
    output_file: Path = None,
    batch_size: int = 32,
    cache: bool = True
) -> np.ndarray:
    """批量提取特征
    
    Args:
        sequences: 序列列表
        method: 嵌入方法
        output_file: 保存路径
        batch_size: 批处理大小
        cache: 是否使用缓存
    
    Returns:
        特征矩阵
    """
    if output_file is None:
        output_file = PROCESSED_DATA_DIR / f"features_{method}.npy"
    
    # 检查缓存
    if cache and output_file.exists():
        print(f"加载缓存特征: {output_file}")
        return np.load(output_file)
    
    # 提取特征
    extractor = get_feature_extractor(method)
    features = extractor._batch_encode(sequences, batch_size)
    
    # 保存
    if cache:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, features)
        print(f"特征已保存: {output_file}")
    
    return features


if __name__ == "__main__":
    # 测试代码
    test_sequences = [
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        "MKTAYIAKVRQGPVKPTKSSVLSQEGCK"
    ]
    
    # 测试 One-Hot
    print("测试 One-Hot 编码...")
    extractor = get_feature_extractor('onehot')
    features = extractor.extract(test_sequences)
    print(f"特征维度: {features.shape}")
    
    # 测试 ESM2 (如果有 GPU)
    if torch.cuda.is_available():
        print("\n测试 ESM2 嵌入...")
        extractor = get_feature_extractor('esm2_8M')
        features = extractor.extract(test_sequences)
        print(f"特征维度: {features.shape}")
