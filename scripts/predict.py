#!/usr/bin/env python3
"""
预测脚本 - 使用训练好的模型进行预测
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import MODELS_DIR, DATASETS_DIR, MODEL_CONFIGS
from src.data.dataset import ProteinDataset
from src.models.multi_task_model import MultiTaskProteinClassifier
from src.data.featurization import get_feature_extractor
from src.data.preprocessing import ProteinDataProcessor


class ProteinClassifierPredictor:
    """蛋白质分类预测器"""
    
    def __init__(
        self,
        model_path: Path,
        embedding_method: str = 'esm2_8M',
        device: str = None,
    ):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.embedding_method = embedding_method
        
        # 加载模型
        self._load_model(model_path)
        
        # 加载编码器
        self._load_encoders()
    
    def _load_model(self, model_path: Path):
        """加载模型"""
        print(f"加载模型: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        raw_args = checkpoint.get('args', None)
        
        # 处理 args，可能是 Namespace 或 dict
        if hasattr(raw_args, '__dict__'):
            self.train_args = vars(raw_args)
        elif isinstance(raw_args, dict):
            self.train_args = raw_args
        else:
            self.train_args = {}
        
        # 先加载编码器获取实际的类别数
        self.processor = ProteinDataProcessor.load_encoders(DATASETS_DIR)
        ec_num_classes = len(self.processor.ec_encoder.classes_)
        loc_num_classes = len(self.processor.loc_encoder.classes_)
        func_num_classes = len(self.processor.func_encoder.classes_)
        
        # 获取输入维度
        embedding_key = self.train_args.get('embedding', self.embedding_method)
        input_dim = MODEL_CONFIGS.get(
            embedding_key,
            MODEL_CONFIGS['esm2_8M']
        )['embedding_dim']
        
        # 创建模型
        self.model = MultiTaskProteinClassifier(
            input_dim=input_dim,
            ec_num_classes=ec_num_classes,
            loc_num_classes=loc_num_classes,
            func_num_classes=func_num_classes,
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型已加载，使用设备: {self.device}")
        print(f"  EC 类别数: {ec_num_classes}")
        print(f"  定位类别数: {loc_num_classes}")
        print(f"  功能类别数: {func_num_classes}")
    
    def _load_encoders(self):
        """加载标签编码器"""
        self.processor = ProteinDataProcessor.load_encoders(DATASETS_DIR)
        self.ec_classes = self.processor.ec_encoder.classes_
        self.loc_classes = self.processor.loc_encoder.classes_
        self.func_classes = self.processor.func_encoder.classes_
    
    @torch.no_grad()
    def predict_single(
        self,
        sequence: str,
        threshold: float = 0.5,
    ) -> Dict:
        """预测单条序列
        
        Args:
            sequence: 蛋白质序列
            threshold: 预测阈值
        
        Returns:
            预测结果字典
        """
        # 提取特征
        extractor = get_feature_extractor(self.embedding_method)
        features = extractor.extract([sequence])
        features = torch.FloatTensor(features).to(self.device)
        
        # 预测
        outputs = self.model(features)
        
        # 处理输出
        ec_probs = torch.sigmoid(outputs['ec']).cpu().numpy()[0]
        loc_probs = torch.softmax(outputs['loc'], dim=1).cpu().numpy()[0]
        func_probs = torch.sigmoid(outputs['func']).cpu().numpy()[0]
        
        # EC 预测 (多标签)
        ec_mask = ec_probs >= threshold
        ec_predictions = [
            (self.ec_classes[i], ec_probs[i])
            for i in range(len(ec_probs))
            if ec_mask[i]
        ]
        ec_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 定位预测 (多分类)
        loc_top3_idx = np.argsort(loc_probs)[-3:][::-1]
        loc_predictions = [
            (self.loc_classes[i], loc_probs[i])
            for i in loc_top3_idx
        ]
        
        # 功能预测 (多标签)
        func_mask = func_probs >= threshold
        func_predictions = [
            (self.func_classes[i], func_probs[i])
            for i in range(len(func_probs))
            if func_mask[i]
        ]
        func_predictions.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'sequence': sequence,
            'sequence_length': len(sequence),
            'ec_predictions': ec_predictions[:5],  # Top 5 EC
            'location_prediction': loc_predictions[0][0],  # Top 1 定位
            'location_top3': loc_predictions,  # Top 3 定位
            'function_predictions': func_predictions[:10],  # Top 10 功能
            'all_predictions': {
                'ec': {self.ec_classes[i]: float(ec_probs[i]) for i in range(len(ec_probs))},
                'location': {self.loc_classes[i]: float(loc_probs[i]) for i in range(len(loc_probs))},
                'function': {self.func_classes[i]: float(func_probs[i]) for i in range(len(func_probs))},
            }
        }
    
    @torch.no_grad()
    def predict_batch(
        self,
        sequences: List[str],
        ids: List[str] = None,
        threshold: float = 0.5,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """批量预测
        
        Args:
            sequences: 蛋白质序列列表
            ids: ID 列表
            threshold: 预测阈值
            batch_size: 批处理大小
        
        Returns:
            预测结果 DataFrame
        """
        if ids is None:
            ids = [f"protein_{i}" for i in range(len(sequences))]
        
        # 提取特征
        extractor = get_feature_extractor(self.embedding_method)
        all_features = extractor._batch_encode(sequences, batch_size)
        
        results = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="预测中"):
            batch_features = torch.FloatTensor(all_features[i:i+batch_size]).to(self.device)
            
            # 预测
            outputs = self.model(batch_features)
            
            # 处理每个样本
            ec_probs = torch.sigmoid(outputs['ec']).cpu().numpy()
            loc_probs = torch.softmax(outputs['loc'], dim=1).cpu().numpy()
            func_probs = torch.sigmoid(outputs['func']).cpu().numpy()
            
            for j in range(len(ec_probs)):
                idx = i + j
                
                # EC 预测
                ec_mask = ec_probs[j] >= threshold
                ec_pred = [self.ec_classes[k] for k in range(len(ec_probs[j])) if ec_mask[k]]
                
                # 定位预测
                loc_pred = self.loc_classes[np.argmax(loc_probs[j])]
                loc_prob = np.max(loc_probs[j])
                
                # 功能预测
                func_mask = func_probs[j] >= threshold
                func_pred = [self.func_classes[k] for k in range(len(func_probs[j])) if func_mask[k]]
                
                results.append({
                    'id': ids[idx],
                    'sequence': sequences[idx],
                    'sequence_length': len(sequences[idx]),
                    'ec_predictions': ','.join(ec_pred[:5]),
                    'ec_count': len(ec_pred),
                    'location_prediction': loc_pred,
                    'location_probability': loc_prob,
                    'function_predictions': ','.join(func_pred[:10]),
                    'function_count': len(func_pred),
                })
        
        return pd.DataFrame(results)
    
    def predict_from_fasta(
        self,
        fasta_path: Path,
        output_path: Path = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """从 FASTA 文件预测
        
        Args:
            fasta_path: FASTA 文件路径
            output_path: 输出文件路径
            threshold: 预测阈值
        
        Returns:
            预测结果 DataFrame
        """
        from Bio import SeqIO
        
        # 读取 FASTA
        records = list(SeqIO.parse(fasta_path, "fasta"))
        sequences = [str(rec.seq) for rec in records]
        ids = [rec.id for rec in records]
        
        print(f"从 FASTA 加载 {len(records)} 条序列")
        
        # 预测
        results = self.predict_batch(sequences, ids, threshold)
        
        # 保存
        if output_path:
            results.to_csv(output_path, index=False, sep='\t')
            print(f"结果已保存: {output_path}")
        
        return results


def parse_args():
    parser = argparse.ArgumentParser(description='蛋白质分类预测')
    
    # 输入
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入序列或 FASTA 文件')
    parser.add_argument('--fasta', action='store_true',
                        help='输入是 FASTA 文件')
    
    # 输出
    parser.add_argument('--output', '-o', type=str, default='predictions.tsv',
                        help='输出文件路径')
    
    # 模型
    parser.add_argument('--model', '-m', type=str,
                        default=str(MODELS_DIR / 'best_model.pt'),
                        help='模型文件路径')
    parser.add_argument('--embedding', '-e', type=str,
                        default='esm2_8M',
                        choices=['onehot', 'esm2_8M', 'esm2_35M', 'esm2_150M', 'protbert'],
                        help='嵌入方法')
    
    # 预测参数
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='预测阈值')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='批处理大小')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建预测器
    predictor = ProteinClassifierPredictor(
        model_path=Path(args.model),
        embedding_method=args.embedding,
    )
    
    if args.fasta:
        # 从 FASTA 预测
        predictor.predict_from_fasta(
            fasta_path=Path(args.input),
            output_path=Path(args.output),
            threshold=args.threshold,
        )
    else:
        # 单条序列预测
        sequence = args.input.strip()
        
        result = predictor.predict_single(sequence, threshold=args.threshold)
        
        print("\n" + "=" * 60)
        print("预测结果")
        print("=" * 60)
        print(f"序列长度: {result['sequence_length']}")
        
        print("\n📊 EC Number 预测 (酶催化功能):")
        if result['ec_predictions']:
            for ec, prob in result['ec_predictions'][:5]:
                print(f"  {ec}: {prob:.4f}")
        else:
            print("  无预测结果")
        
        print(f"\n📍 细胞定位预测:")
        print(f"  主要定位: {result['location_prediction']}")
        print("  Top 3:")
        for loc, prob in result['location_top3']:
            print(f"    {loc}: {prob:.4f}")
        
        print(f"\n🔬 蛋白质功能预测:")
        if result['function_predictions']:
            for func, prob in result['function_predictions'][:5]:
                print(f"  {func}: {prob:.4f}")
        else:
            print("  无预测结果")


if __name__ == "__main__":
    main()
