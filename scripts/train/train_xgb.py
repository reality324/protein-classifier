#!/usr/bin/env python3
"""
Gradient Boosting 训练脚本

功能：训练梯度提升模型预测EC主类
使用sklearn的HistGradientBoostingClassifier，速度更快
"""
import sys
import argparse
import pickle
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.encodings import EncoderRegistry

# 默认配置
DEFAULT_DATA_DIR = "data/datasets/train_subset.parquet"
DEFAULT_ESM2_DIR = "data/processed/esm2_aligned"
DEFAULT_OUTPUT_DIR = "models/xgb"
RANDOM_SEED = 42


def load_data(data_dir, esm2_dir, encoding, test_size=0.2):
    """加载并划分数据"""
    df = pd.read_parquet(data_dir)
    ec_cols = [c for c in df.columns if c.startswith('ec_')]
    y_all = np.argmax(df[ec_cols].values, axis=1)
    
    # 固定随机划分
    indices = np.arange(len(df))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    n = len(df)
    train_end = int(n * (1 - test_size))
    train_idx = indices[:train_end]
    test_idx = indices[train_end:]
    
    # 提取EC主类标签 (1-7 -> 0-6)
    def get_main_class(idx_list):
        labels = np.zeros(len(idx_list), dtype=np.int32)
        for i, idx in enumerate(idx_list):
            labels[i] = int(ec_cols[y_all[idx]].split('_')[1].split('.')[0]) - 1
        return labels
    
    if encoding == "esm2":
        # ESM2已预先划分：train=3000, val=1000, test=1000
        X_train = np.load(esm2_dir / "train_features.npy")
        X_val = np.load(esm2_dir / "val_features.npy")
        X_test = np.load(esm2_dir / "test_features.npy")
        X_train = np.vstack([X_train, X_val])  # 合并train+val
        
        y_all_shuffled = y_all[indices]
        y_train = y_all_shuffled[:4000]
        y_test = y_all_shuffled[4000:]
        
        def to_main_class(labels):
            result = np.zeros(len(labels), dtype=np.int32)
            for i, y in enumerate(labels):
                ec_class = int(ec_cols[y].split('_')[1].split('.')[0]) - 1
                result[i] = ec_class
            return result
        
        y_train = to_main_class(y_train)
        y_test = to_main_class(y_test)
    else:
        encoder = EncoderRegistry.get(encoding)
        X_train = encoder.encode_batch(df['sequence'].iloc[train_idx].tolist())
        X_test = encoder.encode_batch(df['sequence'].iloc[test_idx].tolist())
        y_train = get_main_class(train_idx)
        y_test = get_main_class(test_idx)
    
    return X_train, X_test, y_train, y_test


def train_gb(X_train, y_train, max_iter=100, max_depth=6, learning_rate=0.1):
    """训练HistGradientBoosting模型"""
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    return clf


def main():
    parser = argparse.ArgumentParser(description="训练 HistGradientBoosting 模型")
    parser.add_argument("--encoding", "-e", type=str, default="esm2",
                        choices=["onehot", "ctd", "esm2"], help="特征编码方式")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR, help="数据路径")
    parser.add_argument("--esm2-dir", type=str, default=DEFAULT_ESM2_DIR, help="ESM2特征目录")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--max-iter", type=int, default=100, help="最大迭代次数")
    parser.add_argument("--max-depth", type=int, default=6, help="最大深度")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="学习率")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"训练 HistGradientBoosting 模型 (encoding={args.encoding})")
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
    model = train_gb(X_train, y_train, args.max_iter, args.max_depth, args.learning_rate)
    train_time = time.time() - start_time
    print(f"    训练完成! 耗时: {train_time:.2f}s")
    
    # 评估
    print("\n[3/4] 评估模型...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"    Accuracy: {acc:.4f}, F1 (macro): {f1:.4f}")
    
    # 保存
    print("\n[4/4] 保存模型...")
    model_path = output_dir / f"xgb_{args.encoding}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    config = {
        "algorithm": "HistGradientBoosting",
        "encoding": args.encoding,
        "input_dim": int(X_train.shape[1]),
        "n_classes": 7,
        "max_iter": args.max_iter,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "train_samples": len(y_train),
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1),
        "train_time": float(train_time),
    }
    config_path = output_dir / f"xgb_{args.encoding}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n模型已保存: {model_path}")
    print(f"配置已保存: {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
