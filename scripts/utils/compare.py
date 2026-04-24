#!/usr/bin/env python3
"""compare.py - 编码方式 × 分类算法 对比实验

支持单任务和多任务模式:
- 单任务: EC主类 / Localization / Function
- 多任务: 同时预测 EC + Localization + Function

使用方法:
    # 单任务对比
    python scripts/compare.py --encodings onehot ctd --algorithms rf xgb

    # 多任务对比
    python scripts/compare.py --task multi-task --epochs 30

    # 只对比 EC 主类
    python scripts/compare.py --task ec --encodings ctd --algorithms rf

输出:
    results/
    ├── comparison_results.csv    # 所有实验结果汇总
    ├── models/                   # 保存的模型
    └── plots/                    # 可视化图表
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encodings import EncoderRegistry
from src.algorithms import ClassifierRegistry
from src.pipeline import ProteinDataset, MultiTaskTrainer, Evaluator
from configs.config import MULTITASK_CONFIG, ALL_TASKS


def parse_args():
    parser = argparse.ArgumentParser(description="蛋白质分类器: 编码方式 × 分类算法 对比实验")
    parser.add_argument("--encodings", "-e", nargs="+", default=["onehot", "ctd"],
                        help="指定编码方式")
    parser.add_argument("--algorithms", "-a", nargs="+", default=["rf", "xgb"],
                        help="指定分类算法 (sklearn 系列)")
    parser.add_argument("--task", type=str, default="multi-task",
                        choices=["multi-task", "ec", "localization", "function"],
                        help="任务类型")
    parser.add_argument("--data", "-d", type=str, default=None,
                        help="数据文件 (.parquet)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="神经网络训练轮数")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--output", "-o", type=str,
                        default="results/encoding_algorithm_compare",
                        help="输出目录")
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    return parser.parse_args()


def load_data(data_path: str, task: str, encoding: str):
    """加载数据集"""
    if data_path is None:
        default_paths = [
            Path("data/datasets/train_subset.parquet"),
            Path(__file__).parent.parent / "data" / "datasets" / "train_subset.parquet",
        ]
        for p in default_paths:
            if p.exists():
                data_path = str(p)
                break

    if data_path and Path(data_path).exists():
        print(f"加载数据: {data_path}")
        dataset = ProteinDataset(encoding=encoding, task=task)
        dataset.load_from_parquet(data_path, encoding=encoding)
        return dataset
    else:
        print("未找到数据文件，使用模拟数据演示...")
        np.random.seed(42)
        n_train, n_val, n_test = 1000, 200, 200
        dim = 480
        n_classes = 6

        X_train = np.random.randn(n_train, dim)
        X_val = np.random.randn(n_val, dim)
        X_test = np.random.randn(n_test, dim)
        y_train = np.random.randint(0, n_classes, n_train)
        y_val = np.random.randint(0, n_classes, n_val)
        y_test = np.random.randint(0, n_classes, n_test)

        dataset = ProteinDataset()
        dataset.load_from_arrays(X_train, y_train, X_val, y_val, X_test, y_test)
        return dataset


def run_multitask_comparison(dataset, output_dir, epochs, batch_size, verbose):
    """运行多任务对比"""
    from sklearn.metrics import accuracy_score

    results_list = []

    X_train, y_train_dict = dataset.X_train, dataset.y_train_dict
    X_val, y_val_dict = dataset.X_val, dataset.y_val_dict
    X_test, y_test_dict = dataset.X_test, dataset.y_test_dict

    # 多任务模型对比
    print("\n" + "=" * 60)
    print("多任务学习: 共享编码器 + 3个任务头")
    print("=" * 60)

    # 训练多任务模型
    trainer = MultiTaskTrainer(
        input_dim=X_train.shape[1],
        task_dims={t: MULTITASK_CONFIG["tasks"][t]["num_classes"] for t in ALL_TASKS},
        hidden_dims=[512, 256],
        dropout=0.3,
    )

    print("\n训练多任务模型...")
    trainer.fit(
        X_train, y_train_dict,
        X_val, y_val_dict,
        epochs=epochs,
        batch_size=batch_size,
        patience=10,
        verbose=verbose,
    )

    # 评估
    results = trainer.evaluate(X_test, y_test_dict)
    evaluator = Evaluator()

    print("\n多任务结果:")
    for task in ALL_TASKS:
        task_name = MULTITASK_CONFIG["tasks"][task]["name"]
        print(f"\n【{task_name}】")
        evaluator.print_summary(results[task])

        results_list.append({
            "encoding": "esm2",
            "algorithm": "multitask",
            "task": task,
            "accuracy": results[task].get("accuracy", 0),
            "f1_macro": results[task].get("f1_macro", 0),
        })

    # 保存结果
    df = pd.DataFrame(results_list)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "multitask_results.csv", index=False)

    return df


def visualize_results(results_df: pd.DataFrame, output_dir: Path):
    """可视化对比结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams['font.size'] = 10

    # 按任务分组绘制
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, task in enumerate(ALL_TASKS):
        ax = axes[i]
        task_results = results_df[results_df['task'] == task].sort_values('accuracy', ascending=True)

        if len(task_results) > 0:
            bars = ax.barh(task_results['algorithm'], task_results['accuracy'], color='steelblue', alpha=0.8)
            ax.set_xlabel('Accuracy')
            ax.set_title(f"{task}: {MULTITASK_CONFIG['tasks'][task]['name']}")
            ax.set_xlim(0, 1)
            for bar, acc in zip(bars, task_results['accuracy']):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{acc:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'multitask_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存: {output_dir / 'multitask_comparison.png'}")


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("   蛋白质分类器 - 编码方式 × 分类算法 对比实验")
    print("=" * 70)
    print(f"\n任务: {args.task}")
    print(f"编码方式: {args.encodings}")
    print(f"分类算法: {args.algorithms}")

    # 显示任务信息
    if args.task == "multi-task":
        print("\n多任务配置:")
        for task in ALL_TASKS:
            cfg = MULTITASK_CONFIG["tasks"][task]
            print(f"  - {cfg['name']}: {cfg['num_classes']} 类")
    else:
        cfg = MULTITASK_CONFIG["tasks"][args.task]
        print(f"\n单任务: {cfg['name']} ({cfg['num_classes']} 类)")

    # 加载数据
    dataset = load_data(args.data, args.task, args.encodings[0] if args.encodings else "esm2")

    # 多任务模式
    if args.task == "multi-task":
        results_df = run_multitask_comparison(
            dataset, args.output, args.epochs, args.batch_size, args.verbose
        )
        visualize_results(results_df, Path(args.output))
    else:
        # 单任务模式 (使用现有 sklearn 算法对比)
        print("\n单任务对比暂未实现，请使用 --task multi-task")
        print("或使用 scripts/train_multitask.py 进行单任务训练")


if __name__ == "__main__":
    main()
