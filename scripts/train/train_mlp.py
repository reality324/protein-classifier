#!/usr/bin/env python3
"""
MLP 训练脚本

功能：训练MLP神经网络预测EC主类
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

from src.encodings import EncoderRegistry

# 默认配置
DEFAULT_DATA_DIR = "data/datasets/train_subset.parquet"
DEFAULT_ESM2_DIR = "data/processed/esm2_aligned"
DEFAULT_OUTPUT_DIR = "models/mlp"
RANDOM_SEED = 42


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], n_classes=7, dropout=0.3):
        super().__init__()
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
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_data(data_dir, esm2_dir, encoding, test_size=0.2):
    """加载并划分数据"""
    df = pd.read_parquet(data_dir)
    ec_cols = [c for c in df.columns if c.startswith('ec_')]
    y_all = np.argmax(df[ec_cols].values, axis=1)
    
    indices = np.arange(len(df))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    n = len(df)
    train_end = int(n * (1 - test_size))
    train_idx = indices[:train_end]
    test_idx = indices[train_end:]
    
    def get_main_class(idx_list):
        return np.array([
            int(ec_cols[y_all[i]].split('_')[1].split('.')[0]) - 1 
            for i in idx_list
        ])
    
    if encoding == "esm2":
        X_train = np.load(esm2_dir / "train_features.npy")
        X_test = np.load(esm2_dir / "test_features.npy")
    else:
        encoder = EncoderRegistry.get(encoding)
        X_train = encoder.encode_batch(df['sequence'].iloc[train_idx].tolist())
        X_test = encoder.encode_batch(df['sequence'].iloc[test_idx].tolist())
    
    y_train = get_main_class(train_idx)
    y_test = get_main_class(test_idx)
    
    return X_train, X_test, y_train, y_test


def train_mlp(X_train, y_train, X_test, y_test, hidden_dims=[256, 128], 
              epochs=100, batch_size=64, lr=0.001):
    """训练MLP模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 转Tensor
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    # 模型
    model = MLPClassifier(X_train.shape[1], hidden_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        # Mini-batch
        indices = torch.randperm(len(X_train_t))
        total_loss = 0
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, preds = torch.max(outputs, 1)
            f1 = f1_score(y_test, preds.cpu().numpy(), average='macro')
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Val F1: {f1:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description="训练 MLP 模型")
    parser.add_argument("--encoding", "-e", type=str, default="esm2",
                        choices=["onehot", "ctd", "esm2"], help="特征编码方式")
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
    print(f"训练 MLP 模型 (encoding={args.encoding})")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    X_train, X_test, y_train, y_test = load_data(
        Path(args.data), Path(args.esm2_dir), args.encoding
    )
    print(f"    训练样本: {len(y_train)}, 测试样本: {len(y_test)}")
    print(f"    特征维度: {X_train.shape[1]}")
    
    # 训练
    print("\n[2/4] 训练模型...")
    start_time = time.time()
    model, scaler = train_mlp(X_train, y_train, X_test, y_test, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
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
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"    Accuracy: {acc:.4f}, F1 (macro): {f1:.4f}")
    
    # 保存
    print("\n[4/4] 保存模型...")
    checkpoint = {
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "input_dim": X_train.shape[1],
        "hidden_dims": [256, 128],
        "n_classes": 7,
    }
    model_path = output_dir / f"mlp_{args.encoding}_model.pt"
    torch.save(checkpoint, model_path)
    
    config = {
        "algorithm": "MLP",
        "encoding": args.encoding,
        "input_dim": int(X_train.shape[1]),
        "hidden_dims": [256, 128],
        "n_classes": 7,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_samples": len(y_train),
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1),
        "train_time": float(train_time),
    }
    config_path = output_dir / f"mlp_{args.encoding}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n模型已保存: {model_path}")
    print(f"配置已保存: {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
