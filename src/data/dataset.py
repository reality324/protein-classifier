"""
PyTorch Dataset 实现
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import DATASETS_DIR


class ProteinDataset(Dataset):
    """蛋白质多任务数据集"""
    
    def __init__(
        self,
        data_path: Path = None,
        features: np.ndarray = None,
        ec_labels: np.ndarray = None,
        loc_labels: np.ndarray = None,
        func_labels: np.ndarray = None,
        sequences: List[str] = None,
        ids: List[str] = None,
    ):
        """
        Args:
            data_path: 数据文件路径 (parquet)
            features: 预计算的特征矩阵
            ec_labels: EC 标签 (多标签)
            loc_labels: 定位标签 (单标签)
            func_labels: 功能标签 (多标签)
            sequences: 原始序列
            ids: 蛋白质 ID
        """
        if data_path is not None:
            self._load_from_file(data_path)
        else:
            self.features = features
            self.ec_labels = ec_labels
            self.loc_labels = loc_labels
            self.func_labels = func_labels
            self.sequences = sequences
            self.ids = ids
        
        self.n_samples = len(self.features)
    
    def _load_from_file(self, data_path: Path):
        """从文件加载数据"""
        df = pd.read_parquet(data_path)
        
        self.features = np.stack(df['features'].values)
        self.ids = df['id'].tolist()
        self.sequences = df['sequence'].tolist() if 'sequence' in df.columns else None
        
        # 加载标签
        if 'ec_encoded' in df.columns:
            self.ec_labels = np.stack(df['ec_encoded'].values)
        else:
            self.ec_labels = None
        
        if 'loc_encoded' in df.columns:
            self.loc_labels = df['loc_encoded'].values
        else:
            self.loc_labels = None
        
        if 'func_encoded' in df.columns:
            self.func_labels = np.stack(df['func_encoded'].values)
        else:
            self.func_labels = None
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本"""
        item = {
            'idx': idx,
            'features': torch.FloatTensor(self.features[idx]),
            'id': self.ids[idx],
        }
        
        # 添加标签 (如果有)
        if self.ec_labels is not None:
            item['ec_labels'] = torch.FloatTensor(self.ec_labels[idx])
        
        if self.loc_labels is not None:
            item['loc_labels'] = torch.LongTensor([self.loc_labels[idx]]).squeeze()
        
        if self.func_labels is not None:
            item['func_labels'] = torch.FloatTensor(self.func_labels[idx])
        
        return item


class ProteinDatasetWithEmbedding(Dataset):
    """动态计算嵌入的数据集 (使用预训练模型)"""
    
    def __init__(
        self,
        sequences: List[str],
        ids: List[str],
        embedding_extractor,
        ec_labels: np.ndarray = None,
        loc_labels: np.ndarray = None,
        func_labels: np.ndarray = None,
        batch_size: int = 32,
    ):
        """
        Args:
            sequences: 蛋白质序列列表
            ids: 蛋白质 ID 列表
            embedding_extractor: 嵌入提取器
            ec_labels: EC 标签
            loc_labels: 定位标签
            func_labels: 功能标签
            batch_size: 嵌入计算的批量大小
        """
        self.sequences = sequences
        self.ids = ids
        self.extractor = embedding_extractor
        self.ec_labels = ec_labels
        self.loc_labels = loc_labels
        self.func_labels = func_labels
        
        # 预计算所有嵌入
        print("预计算蛋白质嵌入...")
        self.features = self.extractor._batch_encode(sequences, batch_size)
        self.n_samples = len(self.features)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'idx': idx,
            'features': torch.FloatTensor(self.features[idx]),
            'sequence': self.sequences[idx],
            'id': self.ids[idx],
        }
        
        if self.ec_labels is not None:
            item['ec_labels'] = torch.FloatTensor(self.ec_labels[idx])
        
        if self.loc_labels is not None:
            item['loc_labels'] = torch.LongTensor([self.loc_labels[idx]]).squeeze()
        
        if self.func_labels is not None:
            item['func_labels'] = torch.FloatTensor(self.func_labels[idx])
        
        return item


class DataLoaderFactory:
    """数据加载器工厂"""
    
    @staticmethod
    def create_dataloaders(
        train_path: Path,
        val_path: Path,
        test_path: Path,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> Tuple:
        """创建训练/验证/测试数据加载器"""
        from torch.utils.data import DataLoader
        
        train_dataset = ProteinDataset(train_path)
        val_dataset = ProteinDataset(val_path)
        test_dataset = ProteinDataset(test_path)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return train_loader, val_loader, test_loader


def prepare_data_with_features(
    df: pd.DataFrame,
    features: np.ndarray,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, ProteinDataset]:
    """准备数据集
    
    Args:
        df: 包含标签的 DataFrame
        features: 预计算的特征矩阵
        test_size: 测试集比例
        val_size: 验证集比例
    
    Returns:
        dict: {'train', 'val', 'test'} 数据集
    """
    from sklearn.model_selection import train_test_split
    
    # 分割索引
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    train_val_idx, val_idx = train_test_split(
        train_idx, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    datasets = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        subset = df.iloc[idx].reset_index(drop=True)
        feat_subset = features[idx]
        
        datasets[name] = ProteinDataset(
            features=feat_subset,
            ec_labels=np.stack(subset['ec_encoded'].values) if 'ec_encoded' in subset.columns else None,
            loc_labels=subset['loc_encoded'].values if 'loc_encoded' in subset.columns else None,
            func_labels=np.stack(subset['func_encoded'].values) if 'func_encoded' in subset.columns else None,
            sequences=subset['sequence'].tolist() if 'sequence' in subset.columns else None,
            ids=subset['id'].tolist(),
        )
    
    return datasets


if __name__ == "__main__":
    # 测试 Dataset
    from configs.config import DATASETS_DIR
    
    train_path = DATASETS_DIR / "train.parquet"
    if train_path.exists():
        dataset = ProteinDataset(train_path)
        print(f"训练集样本数: {len(dataset)}")
        
        sample = dataset[0]
        print("样本示例:")
        print(f"  ID: {sample['id']}")
        print(f"  Features shape: {sample['features'].shape}")
        if 'ec_labels' in sample:
            print(f"  EC labels shape: {sample['ec_labels'].shape}")
        if 'loc_labels' in sample:
            print(f"  Location label: {sample['loc_labels']}")
