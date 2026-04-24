#!/usr/bin/env python3
"""
Bayesian Neural Network (BNN) 推理脚本

功能：使用训练好的BNN模型预测EC主类，并提供不确定性估计
"""
import sys
import argparse
import torch
import torch.nn as nn
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

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


class BayesianMLP(nn.Module):
    """带 Dropout 的贝叶斯 MLP (MC Dropout)"""
    def __init__(self, input_dim, hidden_dims=[256, 128], n_classes=7, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def enable_dropout(self):
        """启用 dropout 用于 MC 采样"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


def load_model(model_path):
    """加载模型"""
    model_path = Path(model_path)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = BayesianMLP(
        checkpoint['input_dim'],
        checkpoint.get('hidden_dims', [256, 128]),
        checkpoint.get('n_classes', 7),
        checkpoint.get('dropout', 0.2)
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    
    config_path = model_path.with_suffix('.json')
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = None
    
    return model, scaler, config


def predict(sequence, model, scaler, encoder, n_mc_samples=30):
    """单条预测，带不确定性估计"""
    features = encoder.encode(sequence).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # MC Dropout 采样
    mc_probs = []
    model.enable_dropout()
    with torch.no_grad():
        for _ in range(n_mc_samples):
            logits = model(torch.FloatTensor(features_scaled))
            probs = torch.softmax(logits, dim=1)[0].numpy()
            mc_probs.append(probs)
    
    mc_probs = np.array(mc_probs)
    avg_probs = mc_probs.mean(axis=0)
    pred_class = int(np.argmax(avg_probs))
    
    # 计算不确定性 (预测熵)
    uncertainty = -np.sum(avg_probs * np.log(avg_probs + 1e-8))
    
    return {
        "predicted_class": pred_class,
        "predicted_label": EC_CLASSES.get(pred_class, f"Class {pred_class}"),
        "confidence": float(avg_probs[pred_class]),
        "uncertainty": float(uncertainty),
        "top_predictions": [
            {"class": int(i), "label": EC_CLASSES.get(i, f"Class {i}"), "probability": float(avg_probs[i])}
            for i in np.argsort(avg_probs)[::-1][:3]
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
    parser = argparse.ArgumentParser(description="BNN模型推理")
    parser.add_argument("--model", "-m", type=str, required=True, help="模型路径 (.pt)")
    parser.add_argument("--sequence", "-s", type=str, help="单条蛋白质序列")
    parser.add_argument("--fasta", "-f", type=str, help="FASTA文件路径")
    parser.add_argument("--output", "-o", type=str, help="输出JSON文件路径")
    parser.add_argument("--mc-samples", type=int, default=30, help="MC Dropout采样数")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return 1
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model, scaler, config = load_model(model_path)
    
    if config:
        encoding = config.get('encoding', 'esm2')
        print(f"  算法: {config.get('algorithm', 'BNN')}")
        print(f"  编码: {encoding}")
        print(f"  MC Dropout采样数: {args.mc_samples}")
    else:
        encoding = 'esm2'
    
    # 加载编码器
    print(f"加载编码器: {encoding}")
    encoder = EncoderRegistry.get(encoding)
    
    print("\n" + "=" * 60)
    print("蛋白质EC主类预测 (Bayesian Neural Network)")
    print("=" * 60)
    
    results = []
    
    if args.sequence:
        result = predict(args.sequence, model, scaler, encoder, args.mc_samples)
        result["sequence"] = args.sequence[:50] + "..." if len(args.sequence) > 50 else args.sequence
        result["sequence_length"] = len(args.sequence)
        results.append(result)
        
        print(f"\n序列: {result['sequence']}")
        print(f"长度: {result['sequence_length']} 氨基酸")
        print(f"\n预测结果: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"不确定性 (熵): {result['uncertainty']:.4f}")
        print("\nTop-3 预测:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['label']} ({pred['probability']:.2%})")
    
    elif args.fasta:
        headers, sequences = read_fasta(args.fasta)
        print(f"\n读取 {len(sequences)} 条蛋白质序列")
        
        for i, (header, seq) in enumerate(zip(headers, sequences)):
            result = predict(seq, model, scaler, encoder, args.mc_samples)
            result["header"] = header
            result["sequence_length"] = len(seq)
            results.append(result)
            
            print(f"\n[{i+1}/{len(sequences)}] {header[:50]}")
            print(f"  预测: {result['predicted_label']} ({result['confidence']:.2%})")
            print(f"  不确定性: {result['uncertainty']:.4f}")
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
