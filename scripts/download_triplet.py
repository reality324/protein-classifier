#!/usr/bin/env python3
"""
下载三标签蛋白质数据
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.uniprot_triplet import UniProtTripletDownloader, TripletDatasetAnalyzer


def main():
    print("="*70)
    print("开始下载 UniProt 三标签蛋白质数据")
    print("="*70)

    # 创建下载器
    downloader = UniProtTripletDownloader()

    # 下载数据
    # 参数说明:
    #   max_proteins: 最大数量 (设为 None 下载全部)
    #   min_seq_length: 最小序列长度
    #   max_seq_length: 最大序列长度

    df = downloader.download_triplet_proteins(
        max_proteins=10000,  # 先下载 10000 条测试
        min_seq_length=50,
        max_seq_length=5000,
    )

    print("\n下载完成!")

    # 分析数据
    if len(df) > 0:
        print("\n数据分析:")
        analyzer = TripletDatasetAnalyzer(
            Path("data/raw/triplet_proteins.parquet")
        )
        analyzer.analyze()


if __name__ == "__main__":
    main()
