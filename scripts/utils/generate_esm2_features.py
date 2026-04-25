#!/usr/bin/env python3
"""generate_esm2_features.py - 为数据集生成 ESM2 特征

从 train_subset.parquet 加载蛋白质序列，生成 ESM2 嵌入特征，
确保特征和标签完全对齐。

使用方法:
    python scripts/generate_esm2_features.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置代理
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encodings import EncoderRegistry


def main():
    # 输出目录
    output_dir = Path("data/processed/esm2_aligned")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    parquet_path = "data/datasets/train_subset.parquet"
    print(f"加载数据: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"数据量: {len(df)} 条")

    # 打乱并划分
    indices = np.arange(len(df))
    np.random.seed(42)
    np.random.shuffle(indices)

    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print(f"划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 获取 ESM2 编码器
    print("\n加载 ESM2 模型...")
    encoder = EncoderRegistry.get("esm2")
    print(f"ESM2 模型加载成功，特征维度: {encoder.dim}")

    # 编码所有序列
    all_sequences = df['sequence'].tolist()

    print("\n开始编码序列...")
    features = []

    batch_size = 32
    for i in tqdm(range(0, len(all_sequences), batch_size), desc="编码"):
        batch = all_sequences[i:i+batch_size]
        batch_features = encoder.encode_batch(batch)
        features.append(batch_features)

    features = np.vstack(features)
    print(f"所有特征形状: {features.shape}")

    # 划分特征
    X_train = features[train_idx]
    X_val = features[val_idx]
    X_test = features[test_idx]

    print(f"\n训练集: {X_train.shape}")
    print(f"验证集: {X_val.shape}")
    print(f"测试集: {X_test.shape}")

    # 保存特征
    print(f"\n保存特征到 {output_dir}...")
    np.save(output_dir / "train_features.npy", X_train)
    np.save(output_dir / "val_features.npy", X_val)
    np.save(output_dir / "test_features.npy", X_test)

    # 保存 info
    import json
    info = {
        "source": parquet_path,
        "num_samples": len(df),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "embedding_method": "esm2",
        "embedding_dim": encoder.dim,
    }
    with open(output_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n特征生成完成！")
    print(f"输出目录: {output_dir}")
    print(f"\n使用方法:")
    print(f"  dataset.load_from_esm2_features(features_dir='{output_dir}')")


if __name__ == "__main__":
    main()
