#!/usr/bin/env python3
"""
完整数据准备流程
从 UniProt 下载数据到预处理完成
"""
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASETS_DIR,
    LOGS_DIR
)
from src.data.download import MultiTaskDataDownloader
from src.data.preprocessing import ProteinDataProcessor, split_dataset
from src.data.featurization import get_feature_extractor, batch_extract_features


def step1_download_data(output_file: Path = None) -> pd.DataFrame:
    """步骤1: 下载数据"""
    print("\n" + "=" * 70)
    print("步骤 1: 下载蛋白质数据")
    print("=" * 70)
    
    if output_file is None:
        output_file = RAW_DATA_DIR / "protein_data_raw.parquet"
    
    # 检查是否已存在
    if output_file.exists():
        print(f"数据已存在: {output_file}")
        print("跳过下载步骤...")
        return pd.read_parquet(output_file)
    
    # 下载数据
    downloader = MultiTaskDataDownloader(output_dir=RAW_DATA_DIR)
    df = downloader.create_unified_dataset()
    
    # 保存
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"\n数据已保存: {output_file}")
    
    return df


def step2_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """步骤2: 数据清洗"""
    print("\n" + "=" * 70)
    print("步骤 2: 数据清洗")
    print("=" * 70)
    
    initial_count = len(df)
    
    # 1. 过滤无效序列
    df = df[df['sequence'].notna()].copy()
    df = df[df['sequence'].str.len() >= 10].copy()
    df = df[df['sequence'].str.len() <= 10000].copy()
    
    # 2. 过滤无标签数据
    df = df[
        (df['ec_number'].notna() & (df['ec_number'] != '')) |
        (df['location'].notna() & (df['location'] != '')) |
        (df['keywords'].notna() & (df['keywords'] != ''))
    ].copy()
    
    # 3. 去重
    df = df.drop_duplicates(subset=['id'])
    
    # 4. 清理序列
    df['sequence'] = df['sequence'].str.upper()
    df['sequence'] = df['sequence'].str.replace('\n', '').str.replace(' ', '')
    
    final_count = len(df)
    
    print(f"清洗前: {initial_count} 条")
    print(f"清洗后: {final_count} 条")
    print(f"移除: {initial_count - final_count} 条 ({(initial_count - final_count)/initial_count*100:.1f}%)")
    
    # 统计
    print("\n标签统计:")
    has_ec = (df['ec_number'] != '').sum()
    has_loc = (df['location'] != '').sum()
    has_kw = (df['keywords'] != '').sum()
    print(f"  有 EC 注释: {has_ec} ({has_ec/len(df)*100:.1f}%)")
    print(f"  有定位注释: {has_loc} ({has_loc/len(df)*100:.1f}%)")
    print(f"  有功能注释: {has_kw} ({has_kw/len(df)*100:.1f}%)")
    
    return df


def step3_preprocess(df: pd.DataFrame) -> tuple:
    """步骤3: 预处理"""
    print("\n" + "=" * 70)
    print("步骤 3: 标签编码")
    print("=" * 70)
    
    # 处理编码
    processor = ProteinDataProcessor()
    df = processor.fit_transform(df)
    
    # 保存编码器
    processor.save_encoders(DATASETS_DIR)
    
    # 统计 EC
    print("\nEC 类别统计:")
    print(f"  总类别数: {len(processor.ec_encoder.classes_)}")
    ec_counts = df['ec_encoded'].apply(lambda x: sum(x) if isinstance(x, np.ndarray) else 0)
    print(f"  平均 EC/样本: {ec_counts.mean():.2f}")
    print(f"  最大 EC/样本: {ec_counts.max()}")
    
    # 统计定位
    print("\n定位类别统计:")
    print(f"  总类别数: {len(processor.loc_encoder.classes_)}")
    
    # 统计功能
    print("\n功能类别统计:")
    print(f"  总类别数: {len(processor.func_encoder.classes_)}")
    
    return df, processor


def step4_split_data(df: pd.DataFrame) -> dict:
    """步骤4: 数据划分"""
    print("\n" + "=" * 70)
    print("步骤 4: 数据划分")
    print("=" * 70)
    
    train_df, val_df, test_df = split_dataset(
        df,
        test_size=0.1,
        val_size=0.1,
        random_state=42
    )
    
    # 保存
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(DATASETS_DIR / 'train.parquet', index=False)
    val_df.to_parquet(DATASETS_DIR / 'val.parquet', index=False)
    test_df.to_parquet(DATASETS_DIR / 'test.parquet', index=False)
    
    print(f"\n数据集已保存到: {DATASETS_DIR}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def step5_extract_features(
    dataset_name: str = 'train',
    embedding_method: str = 'onehot',
    overwrite: bool = False
):
    """步骤5: 特征提取 (可选，使用 ESM2 需要 GPU)"""
    print("\n" + "=" * 70)
    print(f"步骤 5: 特征提取 ({embedding_method})")
    print("=" * 70)
    
    # 加载数据
    data_path = DATASETS_DIR / f'{dataset_name}.parquet'
    if not data_path.exists():
        print(f"数据文件不存在: {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    sequences = df['sequence'].tolist()
    
    print(f"序列数量: {len(sequences)}")
    
    # 提取特征
    extractor = get_feature_extractor(embedding_method)
    
    # 保存路径
    output_dir = PROCESSED_DATA_DIR / embedding_method
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}_features.npy'
    
    if output_file.exists() and not overwrite:
        print(f"特征已存在: {output_file}")
        return
    
    print("正在提取特征 (这可能需要较长时间)...")
    features = extractor._batch_encode(sequences, batch_size=32)
    
    # 保存
    np.save(output_file, features)
    print(f"特征已保存: {output_file}")
    print(f"特征维度: {features.shape}")


def main():
    """主函数 - 运行完整流程"""
    print("\n" + "=" * 70)
    print("蛋白质分类器 - 数据准备流程")
    print("=" * 70)
    print(f"开始时间: {datetime.now()}")
    
    # 创建目录
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASETS_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 步骤1: 下载数据
    df = step1_download_data()
    
    # 步骤2: 数据清洗
    df = step2_clean_data(df)
    
    # 步骤3: 预处理
    df, processor = step3_preprocess(df)
    
    # 步骤4: 数据划分
    datasets = step4_split_data(df)
    
    # 步骤5: 特征提取 (可选)
    # 默认使用 onehot, 如需使用 ESM2:
    # step5_extract_features('train', 'esm2_8M')
    # step5_extract_features('val', 'esm2_8M')
    # step5_extract_features('test', 'esm2_8M')
    
    print("\n" + "=" * 70)
    print("数据准备完成!")
    print("=" * 70)
    print(f"\n下一步:")
    print("  1. 运行训练: python scripts/train.py")
    print("  2. 或先提取 ESM2 特征:")
    print("     python scripts/prepare_data.py --extract_features --embedding esm2_8M")
    print(f"\n结束时间: {datetime.now()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='数据准备流程')
    parser.add_argument('--skip_download', action='store_true',
                       help='跳过下载步骤')
    parser.add_argument('--extract_features', action='store_true',
                       help='提取特征')
    parser.add_argument('--embedding', type=str, default='onehot',
                       choices=['onehot', 'esm2_8M', 'esm2_35M', 'esm2_150M', 'protbert'],
                       help='嵌入方法')
    
    args = parser.parse_args()
    
    if args.extract_features:
        # 只提取特征
        for split in ['train', 'val', 'test']:
            step5_extract_features(split, args.embedding, overwrite=True)
    else:
        # 运行完整流程
        main()
