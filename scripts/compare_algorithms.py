#!/usr/bin/env python3
"""
算法对比脚本
比较多种算法在蛋白质分类任务上的表现
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import DATASETS_DIR
from src.data.dataset import ProteinDataset
from src.data.preprocessing import ProteinDataProcessor
from src.models.algorithm_comparison import AlgorithmComparator, AlgorithmFactory


def load_data(data_path: Path, processor: ProteinDataProcessor = None):
    """加载数据"""
    df = pd.read_parquet(data_path)

    # 准备特征和标签
    features = np.stack(df['features'].values)

    # 加载编码器
    if processor is None:
        processor = ProteinDataProcessor.load_encoders(DATASETS_DIR)

    # 解码标签
    ec_labels = np.stack(df['ec_encoded'].values)
    loc_labels = np.stack(df['loc_encoded'].values)
    func_labels = np.stack(df['func_encoded'].values)

    return {
        'features': features,
        'ec_labels': ec_labels,
        'loc_labels': loc_labels,
        'func_labels': func_labels,
        'processor': processor,
    }


def run_comparison(
    algorithms: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = 'multilabel',
    output_dim: int = None,
    output_dir: Path = None,
):
    """运行算法对比"""
    comparator = AlgorithmComparator()

    results = comparator.compare(
        algorithms=algorithms,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type=task_type,
        input_dim=X_train.shape[1],
        output_dim=output_dim,
    )

    # 打印总结
    comparator.print_summary()

    # 保存对比图
    if output_dir:
        comparator.plot_comparison(output_dir / 'algorithm_comparison.png')

    return results, comparator


def main():
    parser = argparse.ArgumentParser(description='算法对比')

    # 数据
    parser.add_argument('--train_data', type=str,
                       default=str(DATASETS_DIR / 'train.parquet'),
                       help='训练数据路径')
    parser.add_argument('--test_data', type=str,
                       default=str(DATASETS_DIR / 'test.parquet'),
                       help='测试数据路径')

    # 算法
    parser.add_argument('--algorithms', '-a', nargs='+',
                       default=['random_forest', 'logistic_regression', 'neural_network', 'bnn'],
                       choices=['random_forest', 'xgboost', 'svm', 'logistic_regression', 'neural_network', 'bnn'],
                       help='要对比的算法')

    # 任务
    parser.add_argument('--task', type=str, default='location',
                       choices=['ec', 'location', 'function'],
                       help='任务类型 (默认为细胞定位)')

    # 参数
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='最大样本数 (用于加速)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='神经网络训练轮数')
    parser.add_argument('--output_dir', '-o', type=str,
                       default=None,
                       help='输出目录')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DATASETS_DIR.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("蛋白质分类 - 算法对比实验")
    print("="*70)
    print(f"时间: {datetime.now()}")
    print(f"任务: {args.task}")
    print(f"算法: {args.algorithms}")

    # 加载数据
    print("\n加载数据...")

    try:
        train_data = load_data(args.train_data)
        test_data = load_data(args.test_data, train_data['processor'])
    except FileNotFoundError as e:
        print(f"\n数据文件不存在: {e}")
        print("\n请先运行数据准备脚本:")
        print("  python scripts/prepare_data.py")
        return

    # 获取指定任务的标签
    task_labels = {
        'ec': 'ec_labels',
        'location': 'loc_labels',
        'function': 'func_labels',
    }

    label_key = task_labels[args.task]
    y_train = train_data[label_key]
    y_test = test_data[label_key]

    # 限制样本数
    X_train = train_data['features'][:args.max_samples]
    y_train = y_train[:args.max_samples]
    X_test = test_data['features'][:min(args.max_samples // 5, len(test_data['features']))]
    y_test = y_test[:min(args.max_samples // 5, len(y_test))]

    print(f"\n数据加载完成:")
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  标签维度: {y_train.shape[1]}")

    # 调整标签形状
    if args.task == 'location':
        task_type = 'multiclass'
        output_dim = y_train.shape[1]
    else:
        task_type = 'multilabel'
        output_dim = y_train.shape[1]

    # 运行对比
    print("\n" + "="*70)
    print("开始对比实验")
    print("="*70)

    results, comparator = run_comparison(
        algorithms=args.algorithms,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type=task_type,
        output_dim=output_dim,
        output_dir=output_dir,
    )

    # 保存结果
    import json
    results_file = output_dir / f'{args.task}_comparison_results.json'

    # 转换 numpy 类型为 Python 类型
    serializable_results = {}
    for algo, metrics in results.items():
        serializable_results[algo] = {}
        for k, v in metrics.items():
            if hasattr(v, 'item'):
                serializable_results[algo][k] = v.item()
            else:
                serializable_results[algo][k] = v

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n结果已保存: {results_file}")
    print(f"对比图已保存: {output_dir / 'algorithm_comparison.png'}")


if __name__ == "__main__":
    main()
