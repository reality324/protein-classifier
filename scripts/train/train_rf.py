#!/usr/bin/env python3
"""
RandomForest 训练脚本

功能：训练RF模型预测EC主类
"""
import sys
import argparse
import pickle
import json
import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.encodings import EncoderRegistry
from src.utils.metrics import print_classification_metrics

# 默认配置
DEFAULT_DATA_DIR = "data/datasets/train_subset.parquet"
DEFAULT_ESM2_DIR = "data/processed/esm2_aligned"
DEFAULT_OUTPUT_DIR = "models/rf"
RANDOM_SEED = 42


def load_data(data_dir, esm2_dir, encoding, test_size=0.2):
    """加载并划分数据
    
    注意: 数据划分必须与 generate_esm2_features.py 保持一致 (60/20/20)
    """
    df = pd.read_parquet(data_dir)
    # 只选择 ec_1, ec_2, ... ec_7 这样的 one-hot 编码列
    ec_cols = [c for c in df.columns if re.match(r'^ec_\d+$', c)]
    y_all = np.argmax(df[ec_cols].values, axis=1)
    
    # 随机划分 - 必须与 generate_esm2_features.py 完全一致！
    indices = np.arange(len(df))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    n = len(df)
    train_end = int(n * 0.6)    # 60%
    val_end = int(n * 0.8)      # 20%
    
    train_idx = indices[:train_end]   # 0-5374 (5374)
    test_idx = indices[val_end:]      # 5374-8957 (3583) - 使用后半部分作为测试
    
    # 编码
    if encoding == "esm2":
        X_train = np.load(esm2_dir / "train_features.npy")
        X_test = np.load(esm2_dir / "test_features.npy")
    else:
        encoder = EncoderRegistry.get(encoding)
        X_train = encoder.encode_batch(df['sequence'].iloc[train_idx].tolist())
        X_test = encoder.encode_batch(df['sequence'].iloc[test_idx].tolist())
    
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    
    return X_train, X_test, y_train, y_test


def train_rf(X_train, y_train, n_estimators=200, max_depth=15, **kwargs):
    """训练RF模型"""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=kwargs.get('min_samples_split', 5),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def main():
    parser = argparse.ArgumentParser(description="训练 RandomForest 模型")
    parser.add_argument("--encoding", "-e", type=str, default="esm2",
                        choices=["onehot", "ctd", "esm2"], help="特征编码方式")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR, help="数据路径")
    parser.add_argument("--esm2-dir", type=str, default=DEFAULT_ESM2_DIR, help="ESM2特征目录")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--n-estimators", type=int, default=200, help="树的数量")
    parser.add_argument("--max-depth", type=int, default=15, help="最大深度")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"训练 RandomForest 模型 (encoding={args.encoding})")
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
    model = train_rf(X_train, y_train, args.n_estimators, args.max_depth)
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
    model_path = output_dir / f"rf_{args.encoding}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # 保存配置
    config = {
        "algorithm": "RandomForest",
        "encoding": args.encoding,
        "input_dim": int(X_train.shape[1]),
        "n_classes": 7,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "train_samples": len(y_train),
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1),
        "train_time": float(train_time),
    }
    config_path = output_dir / f"rf_{args.encoding}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n模型已保存: {model_path}")
    print(f"配置已保存: {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
