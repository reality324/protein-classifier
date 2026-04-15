#!/usr/bin/env python3
"""
数据验证脚本 - 检查数据集的质量和完整性
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import DATASETS_DIR


class DataValidator:
    """数据集验证器"""
    
    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATASETS_DIR
        self.df = None
        self.validation_results = {}
    
    def load_data(self, split: str = 'train') -> pd.DataFrame:
        """加载数据集"""
        data_file = self.data_path / f'{split}.parquet'
        
        if not data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        
        self.df = pd.read_parquet(data_file)
        print(f"加载 {split} 数据集: {len(self.df)} 条记录")
        return self.df
    
    def validate_basic_info(self) -> Dict:
        """验证基本信息"""
        results = {
            'total_samples': len(self.df),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated(subset=['id']).sum(),
        }
        
        # 序列长度统计
        if 'sequence' in self.df.columns:
            seq_lengths = self.df['sequence'].str.len()
            results['sequence_length'] = {
                'min': int(seq_lengths.min()),
                'max': int(seq_lengths.max()),
                'mean': float(seq_lengths.mean()),
                'median': float(seq_lengths.median()),
            }
            
            # 检查异常长度
            short_seqs = (seq_lengths < 10).sum()
            long_seqs = (seq_lengths > 10000).sum()
            results['sequence_length']['short_count'] = int(short_seqs)
            results['sequence_length']['long_count'] = int(long_seqs)
        
        self.validation_results['basic_info'] = results
        return results
    
    def validate_sequences(self) -> Dict:
        """验证蛋白质序列"""
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        
        results = {
            'valid_sequences': 0,
            'invalid_sequences': 0,
            'invalid_examples': [],
        }
        
        invalid_count = 0
        for idx, seq in enumerate(self.df['sequence']):
            seq_upper = seq.upper()
            
            # 检查是否只包含标准氨基酸
            invalid_chars = set(seq_upper) - valid_amino_acids - {'X', 'U', 'O', 'B', 'Z'}
            
            if invalid_chars:
                invalid_count += 1
                if len(results['invalid_examples']) < 5:
                    results['invalid_examples'].append({
                        'id': self.df.iloc[idx]['id'],
                        'invalid_chars': list(invalid_chars),
                    })
        
        results['invalid_sequences'] = invalid_count
        results['valid_sequences'] = len(self.df) - invalid_count
        results['valid_ratio'] = results['valid_sequences'] / len(self.df)
        
        self.validation_results['sequences'] = results
        return results
    
    def validate_labels(self) -> Dict:
        """验证标签"""
        results = {}
        
        # EC Number
        if 'ec_encoded' in self.df.columns:
            ec_labels = self.df['ec_encoded'].apply(lambda x: sum(x) if isinstance(x, np.ndarray) else 0)
            results['ec'] = {
                'samples_with_ec': int((ec_labels > 0).sum()),
                'total_ec_classes': len(ec_labels[ec_labels > 0].unique()),
                'mean_ec_per_sample': float(ec_labels.mean()),
                'max_ec_per_sample': int(ec_labels.max()),
            }
        
        # Location
        if 'loc_encoded' in self.df.columns:
            loc_counts = Counter(self.df['loc_encoded'])
            results['location'] = {
                'total_classes': len(loc_counts),
                'class_distribution': dict(loc_counts.most_common(10)),
            }
        
        # Function
        if 'func_encoded' in self.df.columns:
            func_labels = self.df['func_encoded'].apply(lambda x: sum(x) if isinstance(x, np.ndarray) else 0)
            results['function'] = {
                'samples_with_func': int((func_labels > 0).sum()),
                'total_func_classes': len(func_labels[func_labels > 0].unique()),
                'mean_func_per_sample': float(func_labels.mean()),
            }
        
        self.validation_results['labels'] = results
        return results
    
    def validate_class_balance(self) -> Dict:
        """验证类别平衡"""
        results = {}
        
        # Location 类别平衡
        if 'loc_encoded' in self.df.columns:
            loc_counts = Counter(self.df['loc_encoded'])
            counts = np.array(list(loc_counts.values()))
            
            results['location'] = {
                'min_count': int(counts.min()),
                'max_count': int(counts.max()),
                'imbalance_ratio': float(counts.max() / counts.min()),
                'is_balanced': bool(counts.max() / counts.min() < 10),
            }
        
        self.validation_results['class_balance'] = results
        return results
    
    def print_report(self):
        """打印验证报告"""
        print("\n" + "=" * 70)
        print("数据集验证报告")
        print("=" * 70)
        
        # 基本信息
        if 'basic_info' in self.validation_results:
            info = self.validation_results['basic_info']
            print(f"\n📊 基本信息:")
            print(f"  总样本数: {info['total_samples']}")
            print(f"  重复样本: {info['duplicates']}")
            
            if 'sequence_length' in info:
                sl = info['sequence_length']
                print(f"  序列长度: min={sl['min']}, max={sl['max']}, mean={sl['mean']:.1f}")
                if sl['short_count'] > 0:
                    print(f"    ⚠️  短序列 (<10): {sl['short_count']}")
                if sl['long_count'] > 0:
                    print(f"    ⚠️  长序列 (>10000): {sl['long_count']}")
        
        # 序列验证
        if 'sequences' in self.validation_results:
            seq = self.validation_results['sequences']
            print(f"\n🔬 序列验证:")
            print(f"  有效序列: {seq['valid_sequences']} ({seq['valid_ratio']:.2%})")
            print(f"  无效序列: {seq['invalid_sequences']}")
            if seq['invalid_examples']:
                print(f"  无效示例:")
                for ex in seq['invalid_examples'][:3]:
                    print(f"    {ex['id']}: {ex['invalid_chars']}")
        
        # 标签验证
        if 'labels' in self.validation_results:
            labels = self.validation_results['labels']
            
            print(f"\n🏷️ 标签验证:")
            
            if 'ec' in labels:
                ec = labels['ec']
                print(f"  EC Number:")
                print(f"    有EC注释: {ec['samples_with_ec']}")
                print(f"    EC类别数: {ec['total_ec_classes']}")
                print(f"    平均EC/样本: {ec['mean_ec_per_sample']:.2f}")
            
            if 'location' in labels:
                loc = labels['location']
                print(f"  细胞定位:")
                print(f"    类别数: {loc['total_classes']}")
                print(f"    Top 5 分布:")
                for cls, cnt in list(loc['class_distribution'].items())[:5]:
                    print(f"      Class {cls}: {cnt}")
            
            if 'function' in labels:
                func = labels['function']
                print(f"  蛋白质功能:")
                print(f"    有功能注释: {func['samples_with_func']}")
                print(f"    功能类别数: {func['total_func_classes']}")
        
        # 类别平衡
        if 'class_balance' in self.validation_results:
            balance = self.validation_results['class_balance']
            print(f"\n⚖️ 类别平衡:")
            
            if 'location' in balance:
                loc = balance['location']
                status = "✅ 良好" if loc['is_balanced'] else "⚠️ 不平衡"
                print(f"  细胞定位: {status}")
                print(f"    最大/最小: {loc['imbalance_ratio']:.2f}")
        
        print("\n" + "=" * 70)
    
    def generate_plots(self, output_dir: Path = None):
        """生成可视化图表"""
        if output_dir is None:
            output_dir = self.data_path.parent / 'plots'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 序列长度分布
        ax1 = axes[0, 0]
        seq_lengths = self.df['sequence'].str.len()
        ax1.hist(seq_lengths, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Count')
        ax1.set_title('Sequence Length Distribution')
        ax1.axvline(seq_lengths.median(), color='r', linestyle='--', label=f'Median: {seq_lengths.median():.0f}')
        ax1.legend()
        
        # 2. EC 数量分布
        ax2 = axes[0, 1]
        if 'ec_encoded' in self.df.columns:
            ec_counts = self.df['ec_encoded'].apply(lambda x: sum(x) if isinstance(x, np.ndarray) else 0)
            ec_counts.hist(bins=20, ax=ax2, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Number of EC Annotations')
            ax2.set_ylabel('Count')
            ax2.set_title('EC Number Distribution per Protein')
        
        # 3. 定位类别分布
        ax3 = axes[1, 0]
        if 'loc_encoded' in self.df.columns:
            loc_counts = Counter(self.df['loc_encoded'])
            loc_df = pd.DataFrame.from_dict(loc_counts, orient='index', columns=['count'])
            loc_df = loc_df.sort_values('count', ascending=True).tail(15)
            ax3.barh(range(len(loc_df)), loc_df['count'], color='steelblue')
            ax3.set_yticks(range(len(loc_df)))
            ax3.set_yticklabels([f'Class {i}' for i in loc_df.index])
            ax3.set_xlabel('Count')
            ax3.set_title('Top 15 Location Classes')
        
        # 4. 功能数量分布
        ax4 = axes[1, 1]
        if 'func_encoded' in self.df.columns:
            func_counts = self.df['func_encoded'].apply(lambda x: sum(x) if isinstance(x, np.ndarray) else 0)
            func_counts.hist(bins=20, ax=ax4, edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Number of Function Annotations')
            ax4.set_ylabel('Count')
            ax4.set_title('Function Annotation Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'data_distribution.png', dpi=150)
        print(f"图表已保存: {output_dir / 'data_distribution.png'}")
        
        return output_dir / 'data_distribution.png'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='验证数据集')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='数据集划分')
    parser.add_argument('--data_dir', type=str, default=str(DATASETS_DIR),
                        help='数据目录')
    parser.add_argument('--plot', action='store_true',
                        help='生成可视化图表')
    
    args = parser.parse_args()
    
    validator = DataValidator(Path(args.data_dir))
    validator.load_data(args.split)
    
    validator.validate_basic_info()
    validator.validate_sequences()
    validator.validate_labels()
    validator.validate_class_balance()
    
    validator.print_report()
    
    if args.plot:
        validator.generate_plots()


if __name__ == "__main__":
    main()
