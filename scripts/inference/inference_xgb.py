#!/usr/bin/env python3
"""
XGBoost 推理脚本

功能：使用训练好的XGBoost模型预测EC主类
"""
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.encodings import EncoderRegistry

# EC类别名称（7类）
EC_CLASSES = {
    0: "EC1 - 氧化还原酶 (Oxidoreductases)",
    1: "EC2 - 转移酶 (Transferases)", 
    2: "EC3 - 水解酶 (Hydrolases)",
    3: "EC4 - 裂解酶 (Lyases)",
    4: "EC5 - 异构酶 (Isomerases)",
    5: "EC6 - 连接酶 (Ligases)",
    6: "EC7 - 转运酶 (Translocases)",
}


def find_config_file(model_path):
    """查找配置文件"""
    model_path = Path(model_path)
    # 模型文件名: xgb_esm2_model.json -> 配置: xgb_esm2_config.json
    base_name = model_path.stem.replace('_model', '')
    config_path = model_path.parent / f"{base_name}_config.json"
    if config_path.exists():
        return config_path
    return None


def load_model(model_path):
    """加载模型"""
    model = xgb.XGBClassifier()
    # 支持 .pkl 和 .json 格式
    if str(model_path).endswith('.pkl'):
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model.load_model(str(model_path))
    
    config_path = find_config_file(model_path)
    if config_path:
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = None
    
    return model, config


def predict(sequence, model, encoder):
    """单条预测"""
    features = encoder.encode(sequence).reshape(1, -1)
    
    pred_class = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    
    return {
        "predicted_class": int(pred_class),
        "predicted_label": EC_CLASSES.get(int(pred_class), f"Class {pred_class}"),
        "confidence": float(probs[pred_class]),
        "top_predictions": [
            {"class": int(i), "label": EC_CLASSES.get(i, f"Class {i}"), "probability": float(probs[i])}
            for i in np.argsort(probs)[::-1][:3]
        ]
    }


def read_fasta(fasta_path):
    """读取FASTA文件"""
    headers, sequences = [], []
    with open(fasta_path, 'r') as f:
        current_seq, current_header = [], None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append(''.join(current_seq))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            headers.append(current_header)
            sequences.append(''.join(current_seq))
    return headers, sequences


def main():
    parser = argparse.ArgumentParser(description="XGBoost模型推理")
    parser.add_argument("--model", "-m", type=str, required=True, help="模型路径 (.json)")
    parser.add_argument("--sequence", "-s", type=str, help="单条蛋白质序列")
    parser.add_argument("--fasta", "-f", type=str, help="FASTA文件路径")
    parser.add_argument("--output", "-o", type=str, help="输出JSON文件路径")
    parser.add_argument("--encoding", "-e", type=str, choices=["onehot", "ctd", "esm2"],
                        help="编码方式")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return 1
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model, config = load_model(model_path)
    
    if config:
        encoding = config.get('encoding', 'esm2')
        print(f"  算法: {config.get('algorithm', 'XGBoost')}")
        print(f"  编码: {encoding}")
    else:
        encoding = 'esm2'
    
    # 加载编码器
    print(f"加载编码器: {encoding}")
    encoder = EncoderRegistry.get(encoding)
    
    print("\n" + "=" * 60)
    print("蛋白质EC主类预测 (XGBoost)")
    print("=" * 60)
    
    results = []
    
    if args.sequence:
        result = predict(args.sequence, model, encoder)
        result["sequence"] = args.sequence[:50] + "..." if len(args.sequence) > 50 else args.sequence
        result["sequence_length"] = len(args.sequence)
        results.append(result)
        
        print(f"\n序列: {result['sequence']}")
        print(f"长度: {result['sequence_length']} 氨基酸")
        print(f"\n预测结果: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.2%}")
        print("\nTop-3 预测:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['label']} ({pred['probability']:.2%})")
    
    elif args.fasta:
        headers, sequences = read_fasta(args.fasta)
        print(f"\n读取 {len(sequences)} 条蛋白质序列")
        
        for i, (header, seq) in enumerate(zip(headers, sequences)):
            result = predict(seq, model, encoder)
            result["header"] = header
            result["sequence_length"] = len(seq)
            results.append(result)
            
            print(f"\n[{i+1}/{len(sequences)}] {header[:50]}")
            print(f"  预测: {result['predicted_label']} ({result['confidence']:.2%})")
    else:
        print("错误: 请提供 --sequence 或 --fasta 参数")
        return 1
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_path}")
    
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
