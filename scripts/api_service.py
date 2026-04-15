#!/usr/bin/env python3
"""
API 服务 - 提供 REST API 接口
"""
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from flask import Flask, request, jsonify
from flasgger import Swagger

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import MODELS_DIR
from src.models.multi_task_model import MultiTaskProteinClassifier
from src.data.featurization import get_feature_extractor
from src.data.preprocessing import ProteinDataProcessor


app = Flask(__name__)
swagger = Swagger(app)

# 全局变量
model = None
extractor = None
processor = None
device = None


def load_model(model_path: str, embedding_method: str = 'esm2_8M'):
    """加载模型"""
    global model, extractor, processor, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)

    # 加载模型
    input_dim = checkpoint['args'].get('input_dim', 320)
    model = MultiTaskProteinClassifier(
        input_dim=input_dim,
        ec_num_classes=checkpoint['args'].get('ec_num_classes', 500),
        loc_num_classes=checkpoint['args'].get('loc_num_classes', 30),
        func_num_classes=checkpoint['args'].get('func_num_classes', 50),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 加载特征提取器
    extractor = get_feature_extractor(embedding_method)

    # 加载编码器
    processor = ProteinDataProcessor.load_encoders()

    print(f"模型加载完成，使用设备: {device}")


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    蛋白质分类预测
    ---
    tags:
      - 预测
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            sequence:
              type: string
              description: 蛋白质序列
            threshold:
              type: number
              description: 预测阈值 (默认 0.5)
    responses:
      200:
        description: 预测结果
    """
    data = request.get_json()

    if not data or 'sequence' not in data:
        return jsonify({'error': 'sequence is required'}), 400

    sequence = data['sequence'].strip().upper()
    threshold = data.get('threshold', 0.5)

    if len(sequence) < 10:
        return jsonify({'error': 'sequence too short'}), 400

    # 提取特征
    features = extractor.extract([sequence])
    features = torch.FloatTensor(features).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(features)

    # 处理输出
    ec_probs = torch.sigmoid(outputs['ec']).cpu().numpy()[0]
    loc_probs = torch.softmax(outputs['loc'], dim=1).cpu().numpy()[0]
    func_probs = torch.sigmoid(outputs['func']).cpu().numpy()[0]

    # EC 预测
    ec_mask = ec_probs >= threshold
    ec_predictions = [
        {'ec': processor.ec_encoder.classes_[i], 'probability': float(ec_probs[i])}
        for i in range(len(ec_probs)) if ec_mask[i]
    ]
    ec_predictions.sort(key=lambda x: x['probability'], reverse=True)

    # 定位预测
    loc_top3_idx = loc_probs.argsort()[-3:][::-1]
    loc_predictions = [
        {'location': processor.loc_encoder.classes_[i], 'probability': float(loc_probs[i])}
        for i in loc_top3_idx
    ]

    # 功能预测
    func_mask = func_probs >= threshold
    func_predictions = [
        {'function': processor.func_encoder.classes_[i], 'probability': float(func_probs[i])}
        for i in range(len(func_probs)) if func_mask[i]
    ]
    func_predictions.sort(key=lambda x: x['probability'], reverse=True)

    return jsonify({
        'sequence_length': len(sequence),
        'ec_predictions': ec_predictions[:5],
        'location_predictions': loc_predictions,
        'function_predictions': func_predictions[:10],
    })


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    批量蛋白质分类预测
    ---
    tags:
      - 预测
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            sequences:
              type: array
              items:
                type: string
              description: 蛋白质序列列表
            threshold:
              type: number
              description: 预测阈值
    responses:
      200:
        description: 批量预测结果
    """
    data = request.get_json()

    if not data or 'sequences' not in data:
        return jsonify({'error': 'sequences is required'}), 400

    sequences = [s.strip().upper() for s in data['sequences']]
    threshold = data.get('threshold', 0.5)

    results = []
    for i, seq in enumerate(sequences):
        if len(seq) < 10:
            results.append({'error': 'sequence too short', 'index': i})
            continue

        # 提取特征
        features = extractor.extract([seq])
        features = torch.FloatTensor(features).to(device)

        # 预测
        with torch.no_grad():
            outputs = model(features)

        # 处理输出
        ec_probs = torch.sigmoid(outputs['ec']).cpu().numpy()[0]
        loc_probs = torch.softmax(outputs['loc'], dim=1).cpu().numpy()[0]
        func_probs = torch.sigmoid(outputs['func']).cpu().numpy()[0]

        # 整理结果
        result = {
            'index': i,
            'sequence_length': len(seq),
            'top_ec': processor.ec_encoder.classes_[ec_probs.argmax()],
            'top_location': processor.loc_encoder.classes_[loc_probs.argmax()],
        }
        results.append(result)

    return jsonify({'results': results})


def main():
    import argparse

    parser = argparse.ArgumentParser(description='蛋白质分类器 API 服务')
    parser.add_argument('--model', '-m', type=str,
                       default=str(MODELS_DIR / 'best_model.pt'),
                       help='模型路径')
    parser.add_argument('--embedding', '-e', type=str,
                       default='esm2_8M',
                       help='嵌入方法')
    parser.add_argument('--host', type=str,
                       default='0.0.0.0',
                       help='服务地址')
    parser.add_argument('--port', type=int,
                       default=5000,
                       help='服务端口')

    args = parser.parse_args()

    # 加载模型
    load_model(args.model, args.embedding)

    # 启动服务
    print(f"\n启动 API 服务...")
    print(f"访问地址: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/apidocs")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
