#!/usr/bin/env python3
"""train_multitask.py - 多任务蛋白质分类器训练

使用 ESM2 预计算特征 + 多任务标签

使用方法:
    python scripts/train_multitask.py --encoding esm2
    python scripts/train_multitask.py --task ec
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ProteinDataset, MultiTaskTrainer, Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="多任务蛋白质分类器训练")
    parser.add_argument("--encoding", "-e", type=str, default="esm2",
                        choices=["esm2", "ctd", "onehot"],
                        help="编码方式")
    parser.add_argument("--data", type=str, default="data/datasets/train_subset.parquet",
                        help="Parquet 数据 (用于 CTD/OneHot)")
    parser.add_argument("--esm2-dir", type=str, default="data/processed/esm2_aligned",
                        help="ESM2 特征目录")
    parser.add_argument("--labels-parquet", type=str, default="data/datasets/train_subset.parquet",
                        help="标签 parquet 文件")
    parser.add_argument("--task", "-t", type=str, default="multi-task",
                        choices=["multi-task", "ec", "localization", "function"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", "-o", type=str, default="results/multitask")
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("多任务蛋白质分类器")
    print("=" * 60)

    # 加载数据
    dataset = ProteinDataset(encoding=args.encoding, task=args.task)

    if args.encoding == "esm2":
        dataset.load_from_esm2_features(
            features_dir=args.esm2_dir,
            labels_parquet=args.labels_parquet
        )
    else:
        dataset.load_from_parquet(args.data, encoding=args.encoding)

    # 任务配置
    if args.task == "multi-task":
        tasks = ["ec", "localization", "function"]
        task_dims = {
            "ec": int(dataset.y_train_dict["ec"].max() + 1),
            "localization": int(dataset.y_train_dict["localization"].max() + 1),
            "function": int(dataset.y_train_dict["function"].max() + 1),
        }
        y_train = dataset.y_train_dict
        y_val = dataset.y_val_dict
        y_test = dataset.y_test_dict
    else:
        tasks = [args.task]
        task_dims = {args.task: int(dataset.y_train.max() + 1)}
        y_train = {args.task: dataset.y_train}
        y_val = {args.task: dataset.y_val}
        y_test = {args.task: dataset.y_test}

    task_names = {"ec": "EC主类", "localization": "细胞定位", "function": "分子功能"}

    print(f"\n任务: {tasks}")
    print(f"任务维度: {task_dims}")
    print(f"特征维度: {dataset.X_train.shape[1]}")
    print(f"训练: {len(dataset.X_train)}, 验证: {len(dataset.X_val)}, 测试: {len(dataset.X_test)}")

    # 训练
    trainer = MultiTaskTrainer(
        input_dim=dataset.X_train.shape[1],
        task_dims=task_dims,
        hidden_dims=[512, 256],
        dropout=0.3,
        learning_rate=args.lr,
    )
    print(f"\n设备: {trainer.device}")
    print(f"开始训练 ({args.epochs} epochs)...")

    trainer.fit(
        dataset.X_train, y_train,
        dataset.X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=15,
        verbose=True,
    )

    # 评估
    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)

    evaluator = Evaluator()
    results = trainer.evaluate(dataset.X_test, y_test)

    for task in tasks:
        print(f"\n【{task_names[task]}】")
        evaluator.print_summary(results[task])

    # 保存
    if args.save_model:
        import json
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save(output_dir / "multitask_model.pt")

        save_results = {}
        for task in results:
            save_results[task] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else
                                  [float(x) for x in v] if isinstance(v, list) else v
                                  for k, v in results[task].items()}

        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存: {output_dir}")

    print("\n完成!")


if __name__ == "__main__":
    main()
