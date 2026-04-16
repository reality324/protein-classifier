"""
训练 BNN (贝叶斯神经网络) 多任务蛋白质分类器
支持: EC Number / 细胞定位 / 蛋白质功能 三个任务
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import DATASETS_DIR, MODELS_DIR, LOGS_DIR, MODEL_CONFIGS
from src.data.featurization import get_feature_extractor
from src.models.algorithm_comparison import BayesianNeuralNetworkAlgorithm, AlgorithmConfig

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='训练 BNN 多任务蛋白质分类器')
    
    parser.add_argument('--train_data', type=str, default=str(DATASETS_DIR / 'train.parquet'))
    parser.add_argument('--val_data', type=str, default=str(DATASETS_DIR / 'val.parquet'))
    parser.add_argument('--test_data', type=str, default=str(DATASETS_DIR / 'test.parquet'))
    
    parser.add_argument('--embedding', type=str, default='esm2_8M',
                        choices=['onehot', 'esm2_8M', 'esm2_35M', 'esm2_150M', 'protbert'])
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--prior_sigma', type=float, default=1.0)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--n_samples', type=int, default=10)
    
    parser.add_argument('--save_dir', type=str, default=str(MODELS_DIR / 'bnn'))
    parser.add_argument('--log_dir', type=str, default=str(LOGS_DIR))
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_parquet_data(data_path: str, extractor, device):
    """加载 parquet 数据"""
    df = pd.read_parquet(data_path)
    
    # 提取特征
    features = extractor.extract(df['sequence'].tolist())
    
    # 标签
    ec_labels = np.stack(df['ec_encoded'].values)
    loc_labels = df['loc_encoded'].values.astype(np.int64)
    func_labels = np.stack(df['func_encoded'].values)
    
    return {
        'features': features,
        'ec': ec_labels,
        'loc': loc_labels,
        'func': func_labels,
        'ids': df['id'].tolist()
    }


def create_dataloader(X, y, batch_size, shuffle=False):
    """创建 DataLoader"""
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y) if y.ndim > 1 else torch.LongTensor(y)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, X, y_true, task_type, threshold=0.5):
    """评估模型"""
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    y_pred, uncertainty = model.predict(X, return_uncertainty=True)
    
    if task_type == 'multiclass':
        acc = accuracy_score(y_true, y_pred)
        return {'accuracy': acc, 'uncertainty_mean': uncertainty.mean()}
    else:
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        return {
            'f1': f1, 'precision': prec, 'recall': rec,
            'uncertainty_mean': uncertainty.mean()
        }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 特征维度
    input_dim = MODEL_CONFIGS[args.embedding]['embedding_dim']
    print(f"特征提取: {args.embedding}, 维度: {input_dim}")
    
    # 加载数据
    print("\n加载数据集...")
    extractor = get_feature_extractor(args.embedding)
    
    train_data = load_parquet_data(args.train_data, extractor, device)
    val_data = load_parquet_data(args.val_data, extractor, device)
    test_data = load_parquet_data(args.test_data, extractor, device)
    
    print(f"训练集: {len(train_data['ids'])} 样本")
    print(f"验证集: {len(val_data['ids'])} 样本")
    print(f"测试集: {len(test_data['ids'])} 样本")
    
    # 任务配置
    ec_num_classes = train_data['ec'].shape[1]
    loc_num_classes = int(train_data['loc'].max()) + 1
    func_num_classes = train_data['func'].shape[1]
    
    print(f"\n任务配置:")
    print(f"  EC 类别数: {ec_num_classes}")
    print(f"  定位类别数: {loc_num_classes}")
    print(f"  功能类别数: {func_num_classes}")
    
    # 创建 BNN 模型
    print("\n创建 BNN 模型...")
    
    config_params = {
        'hidden_dims': args.hidden_dims,
        'prior_sigma': args.prior_sigma,
        'dropout_rate': args.dropout_rate,
        'n_samples': args.n_samples,
        'learning_rate': args.lr
    }
    
    # EC 模型
    ec_config = AlgorithmConfig(name='bnn', type='deep_learning', params=config_params)
    ec_model = BayesianNeuralNetworkAlgorithm(
        ec_config, input_dim, ec_num_classes, task_type='multilabel'
    )
    
    # Location 模型
    loc_config = AlgorithmConfig(name='bnn', type='deep_learning', params=config_params)
    loc_model = BayesianNeuralNetworkAlgorithm(
        loc_config, input_dim, loc_num_classes, task_type='multiclass'
    )
    
    # Function 模型
    func_config = AlgorithmConfig(name='bnn', type='deep_learning', params=config_params)
    func_model = BayesianNeuralNetworkAlgorithm(
        func_config, input_dim, func_num_classes, task_type='multilabel'
    )
    
    models = {'ec': ec_model, 'loc': loc_model, 'func': func_model}
    
    print(f"\n模型参数量:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.model.parameters())
        print(f"  {name}: {params:,}")
    
    # 训练循环
    print("\n开始训练...")
    history = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        epoch_start = datetime.now()
        
        # 训练每个任务
        for task_name, model in models.items():
            print(f"\n训练 {task_name} 任务...")
            
            if task_name == 'ec':
                X, y = train_data['features'], train_data['ec']
            elif task_name == 'loc':
                X, y = train_data['features'], train_data['loc']
            else:
                X, y = train_data['features'], train_data['func']
            
            model.fit(X, y, epochs=1, batch_size=args.batch_size)
        
        # 验证
        print("\n验证...")
        val_results = {}
        val_loss = 0
        
        for task_name, model in models.items():
            if task_name == 'ec':
                X, y = val_data['features'], val_data['ec']
                task_type = 'multilabel'
            elif task_name == 'loc':
                X, y = val_data['features'], val_data['loc']
                task_type = 'multiclass'
            else:
                X, y = val_data['features'], val_data['func']
                task_type = 'multilabel'
            
            metrics = evaluate(model, X, y, task_type)
            val_results[task_name] = metrics
            
            # 计算验证损失
            if task_type == 'multiclass':
                val_loss += 1 - metrics['accuracy']
            else:
                val_loss += 1 - metrics['f1']
        
        val_loss /= 3
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        # 打印结果
        print(f"\n验证结果:")
        for task_name, metrics in val_results.items():
            if 'f1' in metrics:
                print(f"  {task_name}: F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, uncertainty={metrics['uncertainty_mean']:.4f}")
            else:
                print(f"  {task_name}: Acc={metrics['accuracy']:.4f}, uncertainty={metrics['uncertainty_mean']:.4f}")
        
        print(f"Epoch 用时: {epoch_time:.1f}s")
        
        # 保存历史
        history.append({
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'val_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in val_results.items()},
            'time': epoch_time
        })
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            for task_name, model in models.items():
                model_path = Path(args.save_dir) / f'best_{task_name}_bnn.pt'
                model.save(model_path)
            print(f"\n✅ 最佳模型已保存 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"\n早停计数器: {patience_counter}/{args.patience}")
        
        if patience_counter >= args.patience:
            print(f"\n早停触发，训练结束")
            break
    
    # 测试
    print("\n" + "="*60)
    print("测试最佳模型...")
    print("="*60)
    
    # 加载最佳模型
    for task_name, model in models.items():
        model_path = Path(args.save_dir) / f'best_{task_name}_bnn.pt'
        if model_path.exists():
            model.load(model_path)
            print(f"已加载: {model_path}")
    
    test_results = {}
    for task_name, model in models.items():
        if task_name == 'ec':
            X, y = test_data['features'], test_data['ec']
            task_type = 'multilabel'
        elif task_name == 'loc':
            X, y = test_data['features'], test_data['loc']
            task_type = 'multiclass'
        else:
            X, y = test_data['features'], test_data['func']
            task_type = 'multilabel'
        
        metrics = evaluate(model, X, y, task_type)
        test_results[task_name] = metrics
    
    print("\n测试结果:")
    for task_name, metrics in test_results.items():
        if 'f1' in metrics:
            print(f"  {task_name}: F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
        else:
            print(f"  {task_name}: Acc={metrics['accuracy']:.4f}")
    
    # 保存结果
    results = {
        'test_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in test_results.items()},
        'history': history,
        'args': vars(args),
        'task_config': {
            'ec_num_classes': ec_num_classes,
            'loc_num_classes': loc_num_classes,
            'func_num_classes': func_num_classes,
        }
    }
    
    results_path = Path(args.log_dir) / 'bnn_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {results_path}")
    print("训练完成!")


if __name__ == "__main__":
    main()
