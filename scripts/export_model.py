#!/usr/bin/env python3
"""
模型导出脚本 - 导出模型用于部署
"""
import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import MODELS_DIR


def export_to_torchscript(model, output_path, example_input):
    """导出为 TorchScript 格式"""
    model.eval()
    
    # 追踪模式
    traced_model = torch.jit.trace(model, example_input)
    
    # 保存
    traced_model.save(str(output_path))
    print(f"TorchScript 模型已保存: {output_path}")


def export_to_onnx(model, output_path, example_input, opset_version=14):
    """导出为 ONNX 格式"""
    model.eval()
    
    # 导出
    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['features'],
        output_names=['ec', 'location', 'function'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'ec': {0: 'batch_size'},
            'location': {0: 'batch_size'},
            'function': {0: 'batch_size'},
        }
    )
    print(f"ONNX 模型已保存: {output_path}")


def export_state_dict(model_path, output_dir):
    """导出模型权重和配置"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 保存配置
    config = {
        'input_dim': checkpoint['args'].get('input_dim', 320),
        'ec_num_classes': checkpoint['args'].get('ec_num_classes', 500),
        'loc_num_classes': checkpoint['args'].get('loc_num_classes', 30),
        'func_num_classes': checkpoint['args'].get('func_num_classes', 50),
        'embedding_method': checkpoint['args'].get('embedding', 'esm2_8M'),
    }
    
    config_path = output_dir / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"配置已保存: {config_path}")
    
    # 保存权重
    weights_path = output_dir / 'model_weights.pt'
    torch.save(checkpoint['model_state_dict'], weights_path)
    print(f"权重已保存: {weights_path}")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='导出模型')

    parser.add_argument('--model', '-m', type=str,
                       default=str(MODELS_DIR / 'best_model.pt'),
                       help='模型路径')
    parser.add_argument('--output_dir', '-o', type=str,
                       default=str(MODELS_DIR / 'exported'),
                       help='输出目录')
    parser.add_argument('--format', '-f', type=str,
                       choices=['torchscript', 'onnx', 'state_dict', 'all'],
                       default='all',
                       help='导出格式')

    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"模型文件不存在: {model_path}")
        return

    print("=" * 60)
    print("模型导出工具")
    print("=" * 60)

    # 加载配置
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('args', {})

    # 创建模型
    from src.models.multi_task_model import MultiTaskProteinClassifier

    input_dim = config.get('input_dim', 320)
    model = MultiTaskProteinClassifier(
        input_dim=input_dim,
        ec_num_classes=config.get('ec_num_classes', 500),
        loc_num_classes=config.get('loc_num_classes', 30),
        func_num_classes=config.get('func_num_classes', 50),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 示例输入
    example_input = torch.randn(1, input_dim)

    # 导出
    if args.format in ['torchscript', 'all']:
        try:
            export_to_torchscript(model, output_dir / 'model.pt', example_input)
        except Exception as e:
            print(f"TorchScript 导出失败: {e}")

    if args.format in ['onnx', 'all']:
        try:
            export_to_onnx(model, output_dir / 'model.onnx', example_input)
        except Exception as e:
            print(f"ONNX 导出失败: {e}")

    if args.format in ['state_dict', 'all']:
        export_state_dict(model_path, output_dir)

    print("\n导出完成!")


if __name__ == "__main__":
    main()
