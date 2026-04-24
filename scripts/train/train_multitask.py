#!/usr/bin/env python3
"""
Multitask 训练脚本

功能：训练多任务神经网络，同时预测EC主类、细胞定位、分子功能
"""
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 默认配置
DEFAULT_DATA_DIR = "data/datasets/train_subset.parquet"
DEFAULT_ESM2_DIR = "data/processed/esm2_aligned"
DEFAULT_OUTPUT_DIR = "models/multitask"
RANDOM_SEED = 42


class MultitaskModel(nn.Module):
    def __init__(self, input_dim, task_dims, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        self.task_dims = task_dims
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        
        self.heads = nn.ModuleDict({
            task: nn.Linear(prev_dim, dim) 
            for task, dim in task_dims.items()
        })
    
    def forward(self, x):
        features = self.shared(x)
        outputs = {task: head(features) for task, head in self.heads.items()}
        return outputs


def load_data(data_dir, esm2_dir):
    """加载多任务数据"""
    df = pd.read_parquet(data_dir)
    
    # EC标签：提取主类编号 (1-7)
    ec_cols = [c for c in df.columns if c.startswith('ec_')]
    y_ec = np.argmax(df[ec_cols].values, axis=1)  # 0-74的类别
    
    # 从列名提取EC主类: ec_1.1 -> 1, ec_2.1 -> 2, etc.
    def get_main_class(idx_list):
        main_classes = []
        for i in idx_list:
            ec_col = ec_cols[y_ec[i]]
            main_class = int(ec_col.split('_')[1].split('.')[0])  # 提取1-7
            main_classes.append(main_class - 1)  # 转为0-6
        return np.array(main_classes)
    
    # Localization: loc_Cytoplasm -> 0, etc.
    loc_cols = [c for c in df.columns if c.startswith('loc_')]
    y_loc = np.argmax(df[loc_cols].values, axis=1)
    
    # Function: func_Cytoplasm -> 0, etc.
    func_cols = [c for c in df.columns if c.startswith('func_')]
    y_func = np.argmax(df[func_cols].values, axis=1)
    
    # 随机划分
    indices = np.arange(len(df))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    n = len(df)
    train_end = int(n * 0.8)
    train_idx = indices[:train_end]
    test_idx = indices[train_end:]
    
    # ESM2特征
    train_features = np.load(esm2_dir / "train_features.npy")
    test_features = np.load(esm2_dir / "test_features.npy")
    
    # 标签
    y_train = {
        'ec': get_main_class(train_idx),
        'localization': y_loc[train_idx],
        'function': y_func[train_idx],
    }
    y_test = {
        'ec': get_main_class(test_idx),
        'localization': y_loc[test_idx],
        'function': y_func[test_idx],
    }
    
    # 标签名称
    class_names = {
        'ec': {i: f"EC{i+1}" for i in range(7)},
        'localization': {i: c.replace('loc_', '') for i, c in enumerate(loc_cols)},
        'function': {i: c.replace('func_', '') for i, c in enumerate(func_cols)},
    }
    
    return train_features, test_features, y_train, y_test, class_names


def train_multitask(X_train, X_test, y_train, y_test, task_dims, 
                    epochs=100, batch_size=64, lr=0.001):
    """训练多任务模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    
    model = MultitaskModel(X_train.shape[1], task_dims).to(device)
    
    # 损失函数
    criteria = {task: nn.CrossEntropyLoss() for task in task_dims}
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_total_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train_t))
        total_loss = 0
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            loss = 0
            for task in task_dims:
                y_task = torch.LongTensor(y_train[task][batch_idx]).to(device)
                loss += criteria[task](outputs[task], y_task)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            f1_total = 0
            for task in task_dims:
                _, preds = torch.max(outputs[task], 1)
                f1 = f1_score(y_test[task], preds.cpu().numpy(), average='macro')
                f1_total += f1
            
            avg_f1 = f1_total / len(task_dims)
            if avg_f1 > best_total_f1:
                best_total_f1 = avg_f1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Avg F1: {avg_f1:.4f}")
    
    model.load_state_dict(best_model_state)
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description="训练多任务模型")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR, help="数据路径")
    parser.add_argument("--esm2-dir", type=str, default=DEFAULT_ESM2_DIR, help="ESM2特征目录")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("训练多任务模型")
    print("=" * 60)
    
    # 任务配置
    task_dims = {
        'ec': 7,           # EC主类
        'localization': 7,  # 细胞定位
        'function': 7,      # 分子功能
    }
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    X_train, X_test, y_train, y_test, class_names = load_data(Path(args.data), Path(args.esm2_dir))
    print(f"    训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    print(f"    任务: {list(task_dims.keys())}")
    
    # 训练
    print("\n[2/4] 训练模型...")
    start_time = time.time()
    model, scaler = train_multitask(X_train, X_test, y_train, y_test, task_dims,
                                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    train_time = time.time() - start_time
    print(f"    训练完成! 耗时: {train_time:.2f}s")
    
    # 评估
    print("\n[3/4] 评估模型...")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_test_scaled = scaler.transform(X_test)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        outputs = model(X_test_t)
    
    results = {}
    for task in task_dims:
        _, preds = torch.max(outputs[task], 1)
        preds = preds.cpu().numpy()
        acc = accuracy_score(y_test[task], preds)
        f1 = f1_score(y_test[task], preds, average='macro')
        results[task] = {'accuracy': acc, 'f1_macro': f1}
        print(f"    {task}: Acc={acc:.4f}, F1={f1:.4f}")
    
    # 保存
    print("\n[4/4] 保存模型...")
    checkpoint = {
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "input_dim": X_train.shape[1],
        "task_dims": task_dims,
        "hidden_dims": [256, 128],
        "class_names": class_names,  # 保存类别名称
    }
    model_path = output_dir / "multitask_model.pt"
    torch.save(checkpoint, model_path)
    
    config = {
        "algorithm": "MultitaskMLP",
        "encoding": "esm2",
        "input_dim": int(X_train.shape[1]),
        "task_dims": task_dims,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_samples": len(y_train['ec']),
        "test_results": results,
        "train_time": float(train_time),
    }
    config_path = output_dir / "multitask_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n模型已保存: {model_path}")
    print(f"配置已保存: {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
