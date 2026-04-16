"""
训练脚本
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

# 项目路径
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    DATASETS_DIR, MODELS_DIR, LOGS_DIR,
    MODEL_CONFIGS, DEFAULT_EMBEDDING,
)
from src.data.dataset import ProteinDataset
from src.models.multi_task_model import (
    MultiTaskProteinClassifier,
    MultiTaskLoss,
    create_model,
)
from src.utils.metrics import Evaluator, MetricTracker

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='训练多任务蛋白质分类器')
    
    # 数据
    parser.add_argument('--train_data', type=str, default=str(DATASETS_DIR / 'train.parquet'))
    parser.add_argument('--val_data', type=str, default=str(DATASETS_DIR / 'val.parquet'))
    parser.add_argument('--test_data', type=str, default=str(DATASETS_DIR / 'test.parquet'))
    
    # 模型
    parser.add_argument('--embedding', type=str, default=DEFAULT_EMBEDDING,
                        choices=['onehot', 'esm2_8M', 'esm2_35M', 'esm2_150M', 'protbert'])
    parser.add_argument('--input_dim', type=int, default=None,
                        help='特征维度，默认从嵌入模型配置获取')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    
    # 任务权重
    parser.add_argument('--ec_weight', type=float, default=1.0)
    parser.add_argument('--loc_weight', type=float, default=1.0)
    parser.add_argument('--func_weight', type=float, default=1.0)
    
    # 其他
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default=str(MODELS_DIR))
    parser.add_argument('--log_dir', type=str, default=str(LOGS_DIR))
    parser.add_argument('--resume', type=str, default=None, help='恢复训练')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """计算类别权重（基于正样本频率）"""
    # labels: (n_samples, n_classes) 多标签 one-hot 矩阵
    # 使用逆频率: weight = neg_samples / pos_samples
    pos_counts = labels.sum(axis=0)  # 每个类别的正样本数
    neg_counts = len(labels) - pos_counts  # 负样本数

    # 避免除零
    pos_counts = np.maximum(pos_counts, 1)

    # 计算权重: 负样本数 / 正样本数
    weights = neg_counts / pos_counts

    # 归一化到 [1, max_weight] 范围，避免权重过大
    weights = np.clip(weights, 1, weights.max())

    return torch.FloatTensor(weights)


def compute_pos_weights(train_df) -> Dict[str, torch.Tensor]:
    """计算所有任务的正样本权重"""
    pos_weights = {}

    # EC 多标签权重
    if 'ec_encoded' in train_df.columns:
        ec_labels = np.stack(train_df['ec_encoded'].values)
        pos_weights['ec'] = compute_class_weights(ec_labels)
        print(f"EC 正样本权重 - 最小值: {pos_weights['ec'].min():.2f}, 最大值: {pos_weights['ec'].max():.2f}")

    # 定位单标签 - 计算类别权重（不平衡数据）
    if 'loc_encoded' in train_df.columns:
        loc_labels = train_df['loc_encoded'].values
        # 统计每个类别的样本数
        unique_classes = np.unique(loc_labels)
        class_counts = np.array([(loc_labels == c).sum() for c in unique_classes])
        # 权重 = 总样本数 / (类别数 × 该类样本数)
        loc_weights = len(loc_labels) / (len(unique_classes) * class_counts)
        # 映射到原始类别顺序
        loc_weight_tensor = torch.ones(int(loc_labels.max()) + 1)
        for idx, cls in enumerate(unique_classes):
            loc_weight_tensor[int(cls)] = loc_weights[idx]
        pos_weights['loc'] = loc_weight_tensor
        print(f"定位类别权重 - 最小: {pos_weights['loc'].min():.2f}, 最大: {pos_weights['loc'].max():.2f}")

    # 功能多标签权重（限制最大值避免梯度爆炸）
    if 'func_encoded' in train_df.columns:
        func_labels = np.stack(train_df['func_encoded'].values)
        func_weights = compute_class_weights(func_labels)
        # 限制最大权重为 50，避免梯度爆炸
        func_weights = torch.clamp(func_weights, 1, 50)
        pos_weights['func'] = func_weights
        print(f"功能正样本权重 - 最小值: {pos_weights['func'].min():.2f}, 最大值: {pos_weights['func'].max():.2f}")

    return pos_weights


def load_data(train_path, val_path, test_path, batch_size, num_workers, embedding_method='onehot'):
    """加载数据集（直接加载预计算的特征）"""
    print("加载预计算的特征...")

    import pandas as pd
    from src.data.dataset import ProteinDataset

    # 特征文件路径
    features_dir = Path(train_path).parent.parent / "processed" / embedding_method
    train_features_path = features_dir / "train_features.npy"
    val_features_path = features_dir / "val_features.npy"
    test_features_path = features_dir / "test_features.npy"

    # 检查特征文件是否存在
    if not train_features_path.exists():
        raise FileNotFoundError(
            f"特征文件不存在: {train_features_path}\n"
            f"请先运行: python scripts/prepare_data.py --extract_features --embedding {embedding_method}"
        )

    # 加载特征
    train_features = np.load(train_features_path)
    val_features = np.load(val_features_path)
    test_features = np.load(test_features_path)

    print(f"加载特征 - Train: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}")

    # 读取元数据
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # 创建数据集
    train_dataset = ProteinDataset(
        features=train_features,
        ec_labels=np.stack(train_df['ec_encoded'].values) if 'ec_encoded' in train_df.columns else None,
        loc_labels=train_df['loc_encoded'].values if 'loc_encoded' in train_df.columns else None,
        func_labels=np.stack(train_df['func_encoded'].values) if 'func_encoded' in train_df.columns else None,
        sequences=train_df['sequence'].tolist() if 'sequence' in train_df.columns else None,
        ids=train_df['id'].tolist(),
    )

    val_dataset = ProteinDataset(
        features=val_features,
        ec_labels=np.stack(val_df['ec_encoded'].values) if 'ec_encoded' in val_df.columns else None,
        loc_labels=val_df['loc_encoded'].values if 'loc_encoded' in val_df.columns else None,
        func_labels=np.stack(val_df['func_encoded'].values) if 'func_encoded' in val_df.columns else None,
        sequences=val_df['sequence'].tolist() if 'sequence' in val_df.columns else None,
        ids=val_df['id'].tolist(),
    )

    test_dataset = ProteinDataset(
        features=test_features,
        ec_labels=np.stack(test_df['ec_encoded'].values) if 'ec_encoded' in test_df.columns else None,
        loc_labels=test_df['loc_encoded'].values if 'loc_encoded' in test_df.columns else None,
        func_labels=np.stack(test_df['func_encoded'].values) if 'func_encoded' in test_df.columns else None,
        sequences=test_df['sequence'].tolist() if 'sequence' in test_df.columns else None,
        ids=test_df['id'].tolist(),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

    return train_loader, val_loader, test_loader


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    metric_tracker: MetricTracker,
    pos_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        features = batch['features'].to(device)
        
        targets = {
            'ec': batch['ec_labels'].to(device),
            'loc': batch['loc_labels'].to(device),
            'func': batch['func_labels'].to(device),
        }
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(features)
        
        # 计算损失（传递正样本权重）
        loss, task_losses = criterion(outputs, targets, pos_weights=pos_weights)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        n_batches += 1
        
        metric_tracker.update(
            losses={'total': loss.item(), **task_losses}
        )
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {
        'total_loss': total_loss / n_batches,
        **{f'{k}_loss': v / n_batches for k, v in task_losses.items()}
    }


@torch.no_grad()
def validate(
    model,
    val_loader,
    criterion,
    device,
    evaluator: Evaluator,
    pos_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """验证"""
    model.eval()
    
    total_loss = 0
    n_batches = 0
    
    for batch in tqdm(val_loader, desc='Validating'):
        features = batch['features'].to(device)
        
        targets = {
            'ec': batch['ec_labels'].to(device),
            'loc': batch['loc_labels'].to(device),
            'func': batch['func_labels'].to(device),
        }
        
        outputs = model(features)
        loss, task_losses = criterion(outputs, targets, pos_weights=pos_weights)
        
        total_loss += loss.item()
        n_batches += 1
        
        # 更新评估器
        evaluator.update(
            outputs['ec'], outputs['loc'], outputs['func'],
            targets['ec'], targets['loc'], targets['func'],
        )
    
    return {
        'total_loss': total_loss / n_batches,
        **{f'{k}_loss': v / n_batches for k, v in task_losses.items()}
    }


@torch.no_grad()
def test(model, test_loader, device, evaluator: Evaluator) -> Dict:
    """测试"""
    model.eval()
    
    for batch in tqdm(test_loader, desc='Testing'):
        features = batch['features'].to(device)
        
        targets = {
            'ec': batch['ec_labels'].to(device),
            'loc': batch['loc_labels'].to(device),
            'func': batch['func_labels'].to(device),
        }
        
        outputs = model(features)
        
        evaluator.update(
            outputs['ec'], outputs['loc'], outputs['func'],
            targets['ec'], targets['loc'], targets['func'],
        )
    
    return evaluator.compute()


def main():
    args = parse_args()
    
    # 设置
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 获取输入维度
    if args.input_dim is None:
        input_dim = MODEL_CONFIGS[args.embedding]['embedding_dim']
    else:
        input_dim = args.input_dim
    
    # 加载数据集获取类别数
    print("获取数据集信息...")
    train_df = pd.read_parquet(args.train_data)
    
    # 获取 ec_encoded 数组长度
    if 'ec_encoded' in train_df.columns:
        ec_sample = train_df['ec_encoded'].iloc[0]
        ec_num_classes = len(ec_sample) if hasattr(ec_sample, '__len__') else int(ec_sample) + 1
    else:
        ec_num_classes = 500
    
    # 获取 loc_encoded 类别数 (单标签)
    if 'loc_encoded' in train_df.columns:
        loc_num_classes = int(train_df['loc_encoded'].max()) + 1
    else:
        loc_num_classes = 30
    
    # 获取 func_encoded 数组长度
    if 'func_encoded' in train_df.columns:
        func_sample = train_df['func_encoded'].iloc[0]
        func_num_classes = len(func_sample) if hasattr(func_sample, '__len__') else int(func_sample) + 1
    else:
        func_num_classes = 50
    
    print(f"输入维度: {input_dim}")
    print(f"EC 类别数: {ec_num_classes}")
    print(f"定位类别数: {loc_num_classes}")
    print(f"功能类别数: {func_num_classes}")

    # 加载数据
    train_loader, val_loader, test_loader = load_data(
        args.train_data, args.val_data, args.test_data,
        args.batch_size, args.num_workers,
        embedding_method=args.embedding
    )

    # 计算正样本权重（处理标签不平衡）
    print("\n计算类别权重...")
    pos_weights = compute_pos_weights(train_df)
    print(f"权重计算完成")

    # 创建模型
    print("\n创建模型...")
    model = create_model(
        input_dim=input_dim,
        ec_num_classes=ec_num_classes,
        loc_num_classes=loc_num_classes,
        func_num_classes=func_num_classes,
        model_type='multi_task',
    )
    model = model.to(device)
    
    # 损失函数
    criterion = MultiTaskLoss(
        task_weights={
            'ec': args.ec_weight,
            'loc': args.loc_weight,
            'func': args.func_weight,
        },
        use_pos_weight=True,
    )
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 评估器
    evaluator = Evaluator(
        ec_classes=ec_num_classes,
        loc_classes=loc_num_classes,
        func_classes=func_num_classes,
    )
    
    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # 训练（传递正样本权重）
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            MetricTracker(['ec', 'loc', 'func']),
            pos_weights=pos_weights
        )
        
        # 验证（传递正样本权重）
        evaluator.reset()
        val_metrics = validate(
            model, val_loader, criterion, device, evaluator,
            pos_weights=pos_weights
        )
        
        # 计算验证集评估指标
        val_results = evaluator.compute()
        
        # 打印
        print(f"\n训练损失: {train_metrics['total_loss']:.4f}")
        print(f"验证损失: {val_metrics['total_loss']:.4f}")
        print(f"  EC Loss: {val_metrics['ec_loss']:.4f}")
        print(f"  Loc Loss: {val_metrics['loc_loss']:.4f}")
        print(f"  Func Loss: {val_metrics['func_loss']:.4f}")
        
        # 更新学习率
        scheduler.step(val_metrics['total_loss'])
        
        # 保存历史
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['total_loss'],
            'val_loss': val_metrics['total_loss'],
            'val_results': val_results,
        }
        history.append(epoch_history)
        
        # 保存最佳模型
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            
            best_model_path = Path(args.save_dir) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'args': args,
            }, best_model_path)
            print(f"✅ 最佳模型已保存: {best_model_path}")
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{args.patience}")
        
        # 早停
        if patience_counter >= args.patience:
            print(f"\n早停触发，训练结束")
            break
    
    # 测试
    print("\n" + "="*60)
    print("测试最佳模型...")
    print("="*60)
    
    checkpoint = torch.load(Path(args.save_dir) / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator.reset()
    test_results = test(model, test_loader, device, evaluator)
    evaluator.print_summary(test_results)
    
    # 保存结果
    results = {
        'test_results': test_results,
        'history': history,
        'args': vars(args),
    }
    
    results_path = Path(args.log_dir) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n结果已保存: {results_path}")
    print("训练完成!")


if __name__ == "__main__":
    main()
