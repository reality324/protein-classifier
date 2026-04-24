#!/usr/bin/env python3
"""predict.py - 使用训练好的模型进行预测

使用方法:
    # 单条序列预测
    python scripts/predict.py --sequence "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL" --model results/ctd_rf.pt

    # 批量预测 (FASTA文件)
    python scripts/predict.py --fasta proteins.fasta --model results/ctd_rf.pt

    # 指定编码方式和算法
    python scripts/predict.py --sequence "YOUR_SEQUENCE" --encoding ctd --algorithm rf
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encodings import EncoderRegistry
from src.algorithms import ClassifierRegistry
from src.pipeline import ProteinDataset
from configs.config import DATASET_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="蛋白质分类器: 预测")
    parser.add_argument("--sequence", "-s", type=str, default=None,
                        help="单条蛋白质序列")
    parser.add_argument("--fasta", "-f", type=str, default=None,
                        help="FASTA 文件路径")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="模型文件路径")
    parser.add_argument("--encoding", "-e", type=str, default="ctd",
                        choices=["onehot", "ctd", "esm2"],
                        help="编码方式")
    parser.add_argument("--algorithm", "-a", type=str, default="rf",
                        choices=["rf", "xgb", "svm", "lr", "mlp", "bnn"],
                        help="分类算法")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--prob", action="store_true",
                        help="输出预测概率")
    return parser.parse_args()


def predict_single(sequence: str, clf, encoder, class_names: list, output_prob: bool = False):
    """预测单条序列"""
    # 编码
    features = encoder.encode(sequence).reshape(1, -1)

    # 预测
    pred_class = clf.predict(features)[0]
    pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"

    result = {
        "sequence": sequence[:50] + "..." if len(sequence) > 50 else sequence,
        "length": len(sequence),
        "predicted_class": int(pred_class),
        "predicted_label": pred_label,
    }

    if output_prob:
        probs = clf.predict_proba(features)[0]
        result["probabilities"] = {
            class_names[i] if i < len(class_names) else f"Class {i}": float(probs[i])
            for i in range(len(probs))
        }

    return result


def read_fasta(fasta_path: str):
    """读取 FASTA 文件"""
    sequences = []
    headers = []

    with open(fasta_path, 'r') as f:
        current_seq = []
        current_header = None

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append(''.join(current_seq))
                    headers.append(current_header)
                current_header = line[1:]  # 去掉 '>'
                current_seq = []
            else:
                current_seq.append(line)

        if current_header is not None:
            sequences.append(''.join(current_seq))
            headers.append(current_header)

    return headers, sequences


def main():
    args = parse_args()

    # 加载模型
    clf = None
    if args.model:
        print(f"加载模型: {args.model}")
        # 从文件加载
        import pickle
        with open(args.model, 'rb') as f:
            clf = pickle.load(f)
    else:
        print(f"创建新分类器: {args.encoding} + {args.algorithm}")
        clf = ClassifierRegistry.get(args.algorithm)

    # 加载编码器
    encoder = EncoderRegistry.get(args.encoding)

    # 类别名称
    class_names = DATASET_CONFIG["class_names"]

    # 预测
    if args.sequence:
        # 单条序列
        result = predict_single(args.sequence, clf, encoder, class_names, args.prob)
        print("\n" + "=" * 50)
        print("预测结果")
        print("=" * 50)
        print(f"序列: {result['sequence']}")
        print(f"长度: {result['length']}")
        print(f"预测类别: {result['predicted_class']} - {result['predicted_label']}")

        if args.prob and 'probabilities' in result:
            print("\n各类别概率:")
            for label, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
                print(f"  {label:20s}: {prob:.4f}")

    elif args.fasta:
        # FASTA 文件
        headers, sequences = read_fasta(args.fasta)

        print(f"读取 {len(sequences)} 条序列")

        results = []
        for i, (header, seq) in enumerate(zip(headers, sequences)):
            result = predict_single(seq, clf, encoder, class_names, args.prob)
            result["header"] = header
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(sequences)}")

        # 输出结果
        print("\n" + "=" * 50)
        print("预测结果汇总")
        print("=" * 50)

        for result in results:
            print(f"\n{result['header']}")
            print(f"  预测: {result['predicted_class']} - {result['predicted_label']}")

        # 保存结果
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存: {output_path}")
    else:
        print("错误: 请提供 --sequence 或 --fasta 参数")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
