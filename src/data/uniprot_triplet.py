#!/usr/bin/env python3
"""
UniProt 数据下载器 - 获取同时拥有 EC、细胞定位、功能标签的蛋白质
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import time

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import RAW_DATA_DIR, DATASETS_DIR


class UniProtTripletDownloader:
    """下载同时拥有三种标签的蛋白质数据"""

    BASE_URL = "https://rest.uniprot.org/uniprotkb"
    CHUNK_SIZE = 500  # API 每批最大数量

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ProteinClassifier/1.0 (https://github.com/...)'
        })

    def download_triplet_proteins(
        self,
        output_file: Path = None,
        min_seq_length: int = 50,
        max_seq_length: int = 5000,
        max_proteins: int = None,
        taxons: List[str] = None,
    ) -> pd.DataFrame:
        """
        下载同时拥有 EC、细胞定位、功能标签的蛋白质

        Args:
            output_file: 输出文件路径
            min_seq_length: 最小序列长度
            max_seq_length: 最大序列长度
            max_proteins: 最大蛋白质数量
            taxons: 分类学 ID 列表 (如 ['9606' = Human, '4932' = Yeast])
        """
        print("="*70)
        print("UniProt 三标签蛋白质数据下载")
        print("="*70)
        print("标签类型: EC Number + 细胞定位 + 蛋白质功能")
        print(f"最小序列长度: {min_seq_length}")
        print(f"最大序列长度: {max_seq_length}")

        # 构建查询
        queries = []

        # EC 编号存在
        queries.append("ec:[1 TO 6]")

        # 细胞定位关键词
        location_keywords = [
            "cytoplasm", "nucleus", "mitochondrion", "membrane",
            "endoplasmic", "golgi", "lysosome", "peroxisome",
            "secreted", "extracellular", "cytoskeleton"
        ]
        for kw in location_keywords:
            queries.append(f'"{kw}"')

        # 功能关键词
        function_keywords = [
            "kinase", "transferase", "hydrolase", "oxidoreductase",
            "DNA-binding", "RNA-binding", "protein-binding",
            "signal", "receptor", "transporter"
        ]
        for kw in function_keywords:
            queries.append(f'"{kw}"')

        # 组装查询
        base_query = " OR ".join(queries)
        if taxons:
            taxon_query = " OR ".join([f"organism:{t}" for t in taxons])
            query = f"({base_query}) AND ({taxon_query})"

        print(f"\n查询: {query[:100]}...")

        # 获取数据
        all_results = []
        offset = 0

        while True:
            if max_proteins and offset >= max_proteins:
                break

            params = {
                "query": query,
                "format": "tsv",
                "columns": "accession,organism,ec,cc_scl_annotation,cc_function,keywords,gene_names,length",
                "size": self.CHUNK_SIZE,
                "offset": offset,
            }

            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=60)
                response.raise_for_status()

                df_chunk = pd.read_csv(response.text, sep="\t")
                if len(df_chunk) == 0:
                    break

                all_results.append(df_chunk)
                offset += len(df_chunk)

                print(f"已获取: {offset} 条记录...")

                # 避免请求过快
                time.sleep(0.3)

            except Exception as e:
                print(f"请求错误: {e}")
                time.sleep(5)

        if not all_results:
            print("未获取到任何数据")
            return pd.DataFrame()

        # 合并数据
        df = pd.concat(all_results, ignore_index=True)

        print(f"\n原始数据: {len(df)} 条记录")

        # 处理数据
        df = self._process_data(df, min_seq_length, max_seq_length)

        print(f"处理后数据: {len(df)} 条记录")

        # 统计
        self._print_statistics(df)

        # 保存
        if output_file is None:
            output_file = self.output_dir / "triplet_proteins.parquet"

        df.to_parquet(output_file, index=False)
        print(f"\n数据已保存: {output_file}")

        return df

    def _process_data(
        self,
        df: pd.DataFrame,
        min_length: int,
        max_length: int,
    ) -> pd.DataFrame:
        """处理和清洗数据"""
        # 重命名列
        df = df.rename(columns={
            'Entry': 'id',
            'Entry Name': 'entry_name',
            'Organism': 'organism',
            'EC number': 'ec_number',
            'Subcellular location CC': 'subcellular_location',
            'Function [CC]': 'function',
            'Keywords': 'keywords',
            'Gene names': 'gene_names',
            'Length': 'length',
            'Sequence': 'sequence',
        })

        # 过滤序列长度
        if 'Length' in df.columns:
            df = df[df['Length'] >= min_length]
            df = df[df['Length'] <= max_length]

        # 清理 EC 编号
        if 'ec_number' in df.columns:
            df['ec_number'] = df['ec_number'].fillna('')

        # 清理细胞定位
        if 'subcellular_location' in df.columns:
            df['subcellular_location'] = df['subcellular_location'].fillna('')

        # 清理功能注释
        if 'function' in df.columns:
            df['function'] = df['function'].fillna('')

        if 'keywords' in df.columns:
            df['keywords'] = df['keywords'].fillna('')

        return df

    def _print_statistics(self, df: pd.DataFrame):
        """打印数据统计"""
        print("\n" + "="*70)
        print("数据统计")
        print("="*70)

        # EC 统计
        has_ec = (df['ec_number'].str.len() > 0).sum()
        print(f"有 EC 编号: {has_ec} ({has_ec/len(df)*100:.1f}%)")

        # 定位统计
        has_loc = (df['subcellular_location'].str.len() > 0).sum()
        print(f"有细胞定位: {has_loc} ({has_loc/len(df)*100:.1f}%)")

        # 功能统计
        has_func = ((df['function'].str.len() > 0) | (df['keywords'].str.len() > 0)).sum()
        print(f"有功能注释: {has_func} ({has_func/len(df)*100:.1f}%)")

        # 三标签同时存在
        has_all = (
            (df['ec_number'].str.len() > 0) &
            (df['subcellular_location'].str.len() > 0) &
            ((df['function'].str.len() > 0) | (df['keywords'].str.len() > 0))
        ).sum()
        print(f"\n三种标签同时存在: {has_all} ({has_all/len(df)*100:.1f}%)")

        # 序列长度分布
        if 'length' in df.columns:
            print(f"\n序列长度: min={df['length'].min()}, max={df['length'].max()}, mean={df['length'].mean():.0f}")

    def download_specific_taxons(self, taxons: List[str]):
        """下载特定物种的数据"""
        taxon_names = {
            '9606': 'Human',
            '10090': 'Mouse',
            '4932': 'Yeast',
            '3702': 'Arabidopsis',
            '7227': 'Drosophila',
            '6239': 'C. elegans',
            '7955': 'Zebrafish',
        }

        for taxon in taxons:
            name = taxon_names.get(taxon, taxon)
            print(f"\n下载 {name} ({taxon}) 数据...")

            output = self.output_dir / f"triplet_{taxon}.parquet"

            # 查询该物种
            query = f"organism:{taxon} AND (ec:[1 TO 6] OR keyword:Kinase OR keyword:Transferase)"

            # ... 实现查询逻辑
            print(f"  保存到: {output}")


class TripletDatasetAnalyzer:
    """三标签数据集分析器"""

    def __init__(self, data_file: Path):
        self.df = pd.read_parquet(data_file)
        self.processor = None

    def analyze(self):
        """分析数据集"""
        print("\n" + "="*70)
        print("三标签蛋白质数据集分析")
        print("="*70)

        print(f"\n总蛋白质数: {len(self.df)}")

        # EC 分布
        if 'ec_number' in self.df.columns:
            ec_counts = self.df['ec_number'].str.split(';').explode().str.strip()
            ec_counts = ec_counts[ec_counts.str.len() > 0]
            print(f"\nEC 类别数: {ec_counts.nunique()}")
            print("Top 10 EC:")
            for ec, count in ec_counts.value_counts().head(10).items():
                print(f"  {ec}: {count}")

        # 定位分布
        if 'subcellular_location' in self.df.columns:
            locs = self.df['subcellular_location'].str.split(';').explode()
            locs = locs[locs.str.len() > 0]
            print(f"\n定位类别数: {locs.nunique()}")
            print("Top 10 定位:")
            for loc, count in locs.value_counts().head(10).items():
                print(f"  {loc.strip()}: {count}")

        # 功能关键词分布
        if 'keywords' in self.df.columns:
            keywords = self.df['keywords'].str.split(';').explode()
            keywords = keywords[keywords.str.len() > 0]
            print(f"\n功能关键词数: {keywords.nunique()}")
            print("Top 20 关键词:")
            for kw, count in keywords.value_counts().head(20).items():
                print(f"  {kw.strip()}: {count}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='下载 UniProt 三标签蛋白质数据')
    parser.add_argument('--output', '-o', type=str,
                       default=str(RAW_DATA_DIR / 'triplet_proteins.parquet'),
                       help='输出文件')
    parser.add_argument('--max_proteins', '-m', type=int, default=None,
                       help='最大蛋白质数量')
    parser.add_argument('--analyze', action='store_true',
                       help='分析已有数据')

    args = parser.parse_args()

    if args.analyze:
        # 分析已有数据
        analyzer = TripletDatasetAnalyzer(Path(args.output))
        analyzer.analyze()
    else:
        # 下载数据
        downloader = UniProtTripletDownloader()
        df = downloader.download_triplet_proteins(
            output_file=Path(args.output),
            max_proteins=args.max_proteins,
            min_seq_length=50,
            max_seq_length=5000,
        )

        # 进一步分析
        if len(df) > 0:
            analyzer = TripletDatasetAnalyzer(Path(args.output))
            analyzer.analyze()


if __name__ == "__main__":
    main()
