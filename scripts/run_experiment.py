#!/usr/bin/env python3
"""
完整实验启动脚本
一键运行: 下载数据 → 预处理 → 训练 → 评估
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """运行命令并打印结果"""
    print("\n" + "=" * 70)
    print(f"执行: {description}")
    print("=" * 70)
    print(f"命令: {cmd}\n")

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(Path(__file__).parent.parent)
    )

    if result.returncode != 0:
        print(f"❌ {description} 失败!")
        return False
    else:
        print(f"✅ {description} 完成")
        return True


def main():
    parser = argparse.ArgumentParser(description='完整实验流程')

    # 步骤选择
    parser.add_argument('--skip_download', action='store_true', help='跳过下载')
    parser.add_argument('--skip_preprocess', action='store_true', help='跳过预处理')
    parser.add_argument('--skip_train', action='store_true', help='跳过训练')

    # 配置
    parser.add_argument('--embedding', type=str, default='onehot',
                       choices=['onehot', 'esm2_8M', 'esm2_35M', 'esm2_150M'],
                       help='特征提取方法')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    start_time = datetime.now()

    print("\n" + "=" * 70)
    print("蛋白质分类器 - 完整实验流程")
    print("=" * 70)
    print(f"开始时间: {start_time}")
    print(f"嵌入方法: {args.embedding}")
    print(f"训练轮数: {args.epochs}")

    success = True

    # 步骤1: 准备数据
    if not args.skip_download and not args.skip_preprocess:
        success &= run_command(
            'python scripts/prepare_data.py',
            '数据下载和预处理'
        )

    # 步骤2: 训练
    if not args.skip_train:
        train_cmd = f"""
        python scripts/train.py
            --embedding {args.embedding}
            --epochs {args.epochs}
            --batch_size {args.batch_size}
            --lr {args.lr}
            --patience 10
        """.replace('\n', ' ')

        success &= run_command(
            train_cmd,
            '模型训练'
        )

    # 步骤3: 验证数据
    success &= run_command(
        'python scripts/validate_data.py --split train --plot',
        '验证数据集'
    )

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    print(f"开始时间: {start_time}")
    print(f"结束时间: {end_time}")
    print(f"总耗时: {duration}")

    if success:
        print("\n✅ 所有步骤执行成功!")
    else:
        print("\n⚠️ 部分步骤执行失败，请检查日志")


if __name__ == "__main__":
    main()
