#!/usr/bin/env python3
"""
下载小型演示数据集
从 UniProt API 获取少量标注数据用于演示和测试
"""
import os
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import RAW_DATA_DIR


class DemoDataDownloader:
    """演示数据下载器 - 获取小批量高质量标注数据"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ProteinClassifier-Demo/1.0'
        })
    
    def fetch_proteins_from_uniprot(self, organism: str, max_results: int = 500) -> List[Dict]:
        """从 UniProt 获取蛋白质数据 - 使用流式下载"""
        print(f"获取 {organism} 的蛋白质...")
        
        output_file = self.output_dir / f"uniprot_{organism}.json.gz"
        
        if output_file.exists():
            print(f"  文件已存在: {output_file}")
            return self._parse_stream_file(output_file)
        
        # 使用流式 API 获取数据
        url = f"https://rest.uniprot.org/uniprotkb/stream?query=organism:{organism}+AND+reviewed:true&format=json&size=500"
        
        try:
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # 保存到文件
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"  下载完成: {output_file}")
            return self._parse_stream_file(output_file)
            
        except Exception as e:
            print(f"  获取失败: {e}")
            if output_file.exists():
                os.remove(output_file)
            return []
    
    def _parse_stream_file(self, filepath: Path) -> List[Dict]:
        """解析流式下载的文件"""
        import gzip
        results = []
        
        try:
            with gzip.open(filepath, 'rt') as f:
                data = json.load(f)
                for entry in data.get('results', []):
                    parsed = self.parse_protein_entry(entry)
                    if parsed:
                        results.append(parsed)
        except Exception as e:
            print(f"  解析失败: {e}")
        
        return results
    
    def parse_protein_entry(self, entry: Dict) -> Optional[Dict]:
        """解析蛋白质条目"""
        try:
            # 处理新版 API 格式
            if 'entries' in entry:  # 兼容旧格式
                entry = entry['entries'][0] if entry.get('entries') else entry
            
            # 获取序列
            seq_data = entry.get('sequence', {})
            seq = seq_data.get('value', '') if isinstance(seq_data, dict) else str(seq_data)
            
            if len(seq) < 30 or len(seq) > 5000:
                return None
            
            accession = entry.get('primaryAccession', entry.get('accession', ''))
            
            # 基因名
            genes = entry.get('genes', [])
            gene_name = ''
            if genes and isinstance(genes[0], dict):
                gene_name = genes[0].get('geneName', {}).get('value', '')
            
            # 物种
            organism_data = entry.get('organism', {})
            organism = organism_data.get('scientificName', '') if isinstance(organism_data, dict) else str(organism_data)
            
            # EC numbers
            ec_numbers = []
            for item in entry.get('ecNumbers', []):
                if isinstance(item, dict):
                    ec = item.get('value', '')
                else:
                    ec = str(item)
                if '.' in ec and ec.count('.') >= 2:
                    ec_numbers.append(ec)
            
            # GO terms
            go_terms = []
            for item in entry.get('goTerms', []):
                if isinstance(item, dict):
                    go_id = item.get('id', '')
                    go_name = item.get('value', '')
                    go_ont = item.get('ontology', '')
                else:
                    go_id = str(item)
                    go_name = ''
                    go_ont = ''
                if go_id:
                    go_terms.append(f"{go_id}:{go_name}:{go_ont}")
            
            # Subcellular location
            locations = []
            for comment in entry.get('comments', []):
                if isinstance(comment, dict) and comment.get('commentType') == 'SUBCELLULAR LOCATION':
                    for loc in comment.get('subcellularLocations', []):
                        loc_val = loc.get('location', {})
                        if isinstance(loc_val, dict):
                            loc_val = loc_val.get('value', '')
                        if loc_val:
                            locations.append(str(loc_val))
            
            return {
                'id': accession,
                'sequence': seq,
                'gene_name': gene_name,
                'organism': organism,
                'ec_number': ','.join(ec_numbers[:5]),
                'go_terms': ','.join(go_terms[:10]),
                'location': '; '.join(locations[:3]),
                'length': len(seq)
            }
        except Exception as e:
            return None
    
    def download_demo_dataset(self, n_proteins: int = 2000) -> pd.DataFrame:
        """下载演示数据集"""
        print("=" * 60)
        print("开始下载演示数据集...")
        print("=" * 60)
        
        # 选择多个模式生物以获得多样性
        organisms = [
            'Human',           # 人类
            'Escherichia coli',  # 大肠杆菌
            'Drosophila melanogaster',  # 果蝇
            'Mus musculus',     # 小鼠
            'Saccharomyces cerevisiae',  # 酵母
            'Arabidopsis thaliana',    # 拟南芥
        ]
        
        all_proteins = []
        per_species = max(100, n_proteins // len(organisms))
        
        for org in organisms:
            print(f"\n获取 {org} 蛋白质...")
            entries = self.fetch_proteins_from_uniprot(org, per_species)
            all_proteins.extend(entries)
            print(f"  成功获取 {len(entries)} 条")
            
            if len(all_proteins) >= n_proteins:
                break
        
        if len(all_proteins) < 100:
            print("\n真实数据获取不足，使用模拟数据...")
            return self._generate_synthetic_data(n_proteins)
        
        df = pd.DataFrame(all_proteins[:n_proteins])
        
        print(f"\n成功获取 {len(df)} 条蛋白质数据")
        return df
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """生成合成数据用于演示"""
        print("生成合成演示数据...")
        
        np.random.seed(42)
        
        # 常见氨基酸
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # 常见 EC 编号
        ec_prefixes = [
            '1.1.1', '1.2.1', '1.3.1', '1.4.1', '1.11.1',
            '2.1.1', '2.3.1', '2.5.1', '2.7.1', '2.7.11',
            '3.1.1', '3.2.1', '3.4.1', '3.5.1', '3.6.1',
            '4.1.1', '4.2.1', '4.3.1', '4.6.1',
            '5.1.1', '5.2.1', '5.3.1', '5.4.1', '5.5.1',
            '6.1.1', '6.2.1', '6.3.1', '6.4.1'
        ]
        
        # 细胞定位
        locations = [
            'Cytoplasm', 'Nucleus', 'Mitochondrion', 'Endoplasmic reticulum',
            'Golgi apparatus', 'Plasma membrane', 'Extracellular', 'Lysosome',
            'Peroxisome', 'Endosome', 'Cytoskeleton', 'Vacuole'
        ]
        
        # 功能关键词
        functions = [
            'Kinase', 'Transferase', 'Hydrolase', 'Oxidoreductase', 'Ligase',
            'Transcription', 'Binding', 'Transport', 'Structural', 'Signaling',
            'Catalytic', 'DNA-binding', 'RNA-binding', 'ATP-binding', 'Metal-binding'
        ]
        
        data = []
        for i in range(n_samples):
            length = np.random.randint(50, 800)
            seq = ''.join(np.random.choice(list(amino_acids), length))
            
            # 随机选择 EC
            n_ec = np.random.randint(0, 4)
            ec_list = np.random.choice(ec_prefixes, n_ec)
            ec_list = [f"{ec}.{np.random.randint(1,99)}" for ec in ec_list]
            
            n_loc = 1 if np.random.random() > 0.2 else 0
            loc_list = np.random.choice(locations, n_loc) if n_loc else []
            
            n_func = np.random.randint(1, 5)
            func_list = np.random.choice(functions, n_func)
            
            data.append({
                'id': f'SYNTH_{i:05d}',
                'sequence': seq,
                'gene_name': f'GENE_{i}',
                'organism': 'Synthetic',
                'ec_number': ','.join(ec_list),
                'go_terms': ';'.join(func_list),
                'location': ';'.join(loc_list),
                'length': length
            })
        
        df = pd.DataFrame(data)
        print(f"生成 {len(df)} 条合成数据")
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'demo_proteins.parquet'):
        """保存数据集"""
        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False)
        print(f"\n数据集已保存: {output_path}")
        return output_path
    
    def print_statistics(self, df: pd.DataFrame):
        """打印数据集统计"""
        print("\n" + "=" * 60)
        print("数据集统计")
        print("=" * 60)
        print(f"总蛋白质数: {len(df)}")
        print(f"有 EC 注释: {(df['ec_number'].str.len() > 0).sum()}")
        print(f"有定位注释: {(df['location'].str.len() > 0).sum()}")
        print(f"有功能注释: {(df['go_terms'].str.len() > 0).sum()}")
        
        print(f"\n序列长度统计:")
        print(f"  平均: {df['length'].mean():.1f}")
        print(f"  最小: {df['length'].min()}")
        print(f"  最大: {df['length'].max()}")
        
        if 'organism' in df.columns:
            print(f"\n物种分布:")
            print(df['organism'].value_counts().head(10))


def main():
    parser = argparse.ArgumentParser(description='下载演示数据集')
    parser.add_argument('--n', type=int, default=2000, help='蛋白质数量')
    parser.add_argument('--output', type=str, default='demo_proteins.parquet', help='输出文件')
    parser.add_argument('--synthetic', action='store_true', help='使用合成数据')
    
    args = parser.parse_args()
    
    downloader = DemoDataDownloader()
    
    if args.synthetic:
        df = downloader._generate_synthetic_data(args.n)
    else:
        df = downloader.download_demo_dataset(args.n)
    
    downloader.save_dataset(df, args.output)
    downloader.print_statistics(df)
    
    return df


if __name__ == "__main__":
    main()
