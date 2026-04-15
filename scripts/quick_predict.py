#!/usr/bin/env python3
"""
快速推理脚本 - 命令行使用
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict import ProteinClassifierPredictor
import argparse


def main():
    parser = argparse.ArgumentParser(description='蛋白质分类器 - 快速预测')
    parser.add_argument('sequence', type=str, help='蛋白质序列')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='模型路径')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='预测阈值')
    
    args = parser.parse_args()
    
    # 创建预测器
    model_path = args.model or 'models/best_model.pt'
    predictor = ProteinClassifierPredictor(model_path=model_path)
    
    # 预测
    result = predictor.predict_single(args.sequence, threshold=args.threshold)
    
    # 打印结果
    print("\n" + "="*60)
    print("预测结果")
    print("="*60)
    print(f"序列长度: {result['sequence_length']} aa")
    
    print("\n🔬 EC 编号预测:")
    if result['ec_predictions']:
        for i, (ec, prob) in enumerate(result['ec_predictions'][:3], 1):
            print(f"  {i}. {ec} (概率: {prob:.3f})")
    else:
        print("  无预测结果")
    
    print("\n📍 细胞定位:")
    print(f"  预测: {result['location_prediction']}")
    print("  Top 3:")
    for i, (loc, prob) in enumerate(result['location_top3'], 1):
        print(f"    {i}. {loc} (概率: {prob:.3f})")
    
    print("\n🏷️ 蛋白质功能:")
    if result['function_predictions']:
        for i, (func, prob) in enumerate(result['function_predictions'][:3], 1):
            print(f"  {i}. {func} (概率: {prob:.3f})")
    else:
        print("  无预测结果")


if __name__ == "__main__":
    main()