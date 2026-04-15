#!/usr/bin/env python3
"""
评估脚本 - 对测试集进行评估
"""
import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import MODELS_DIR, DATASETS_DIR
from src.data.dataset import ProteinDataset
from src.models.multi_task_model import MultiTaskProteinClassifier
from src.utils.metrics import Evaluator


def load_model(model_path, device='cuda'):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device)

    model = MultiTaskProteinClassifier(
        input_dim=checkpoint['args'].get('input_dim', 320),
        ec_num_classes=checkpoint['args'].get('ec_num_classes', 500),
        loc_num_classes=checkpoint['args'].get('loc_num_classes', 30),
        func_num_classes=checkpoint['args'].get('func_num_classes', 50),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint.get('args', {})


def evaluate(model, dataloader, device='cuda', threshold=0.5):
    """评估模型"""
    evaluator = Evaluator(
        ec_classes=model.ec_head.output.out_features,
        loc_classes=model.loc_head.output.out_features,
        func_classes=model.func_head.output.out_features,
        ec_threshold=threshold,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
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
    parser = argparse.ArgumentParser(description='评估蛋白质分类器')

    # 模型和数据
    parser.add_argument('--model', '-m', type=str,
                        default=str(MODELS_DIR / 'best_model.pt'),
                        help='模型路径')
    parser.add_argument('--test_data', '-t', type=str,
                        default=str(DATASETS_DIR / 'test.parquet'),
                        help='测试数据路径')
    parser.add_argument('--output', '-o', type=str,
                        default=None,
                        help='输出结果路径')

    # 参数
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='预测阈值')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return

    # 检查数据文件
    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"❌ 测试数据不存在: {test_path}")
        return

    print("=" * 70)
    print("蛋白质分类器 - 模型评估")
    print("=" * 70)
    print(f"模型: {model_path}")
    print(f"数据: {test_path}")
    print(f"阈值: {args.threshold}")

    # 加载模型
    print("\n加载模型...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, model_args = load_model(model_path, device)
    print(f"模型加载完成，设备: {device}")

    # 加载数据
    print("\n加载数据...")
    from torch.utils.data import DataLoader

    test_dataset = ProteinDataset(test_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    print(f"测试集: {len(test_dataset)} 样本")

    # 评估
    print("\n开始评估...")
    results = evaluate(model, test_loader, device, args.threshold)

    # 打印结果
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)

    evaluator = Evaluator(1, 1, 1)  # 用于打印
    evaluator.print_summary(results)

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n结果已保存: {output_path}")

    print("\n评估完成!")


if __name__ == "__main__":
    main()
