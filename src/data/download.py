"""
从 UniProt 下载蛋白质数据
支持下载有 EC 注释、功能注释和细胞定位注释的蛋白质
"""
import os
import gzip
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import RAW_DATA_DIR, UNIPROT_URLS


class UniProtDownloader:
    """UniProt 数据下载器"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ProteinClassifier/1.0 (Research Purpose)'
        })
    
    def download_file(self, url: str, output_file: Path) -> bool:
        """下载单个文件"""
        if output_file.exists():
            print(f"文件已存在: {output_file}")
            return True
        
        try:
            print(f"正在下载: {url}")
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"下载完成: {output_file}")
            return True
        except Exception as e:
            print(f"下载失败: {e}")
            if output_file.exists():
                os.remove(output_file)
            return False
    
    def download_sprot(self) -> Path:
        """下载 SwissProt 数据库"""
        output_file = self.output_dir / "uniprot_sprot.dat.gz"
        self.download_file(UNIPROT_URLS["sprot"], output_file)
        return output_file
    
    def parse_dat_file(self, dat_file: Path) -> pd.DataFrame:
        """解析 UniProt .dat 文件

        提取: ID, 序列, EC号, GO terms, 细胞定位
        """
        import re
        
        records = []
        current_record = {}
        
        with gzip.open(dat_file, 'rt') as f:
            for line in f:
                line = line.strip()
                
                # 记录开始/结束
                if line.startswith('ID   '):
                    if current_record:
                        records.append(current_record)
                    current_record = {'ID': line[5:].strip()}
                    
                elif line.startswith('AC   '):
                    current_record['AC'] = line[5:].strip()
                    
                elif line.startswith('SQ   '):
                    # 序列开始标记
                    pass

                elif line.startswith('    ') and 'sequence' not in current_record:
                    # 序列行（缩进，不含空格的其他内容）
                    seq = line.strip().replace(' ', '')
                    current_record['sequence'] = seq
                    
                elif line.startswith('DE   '):
                    if 'description' not in current_record:
                        current_record['description'] = ''
                    current_record['description'] += line[5:].strip() + ' '
                    
                elif line.startswith('GN   '):
                    current_record['gene_name'] = line[5:].strip()
                    
                elif line.startswith('OS   '):
                    current_record['organism'] = line[5:].strip()
                    
                elif line.startswith('OC   '):
                    if 'organism_class' not in current_record:
                        current_record['organism_class'] = ''
                    current_record['organism_class'] += line[5:].strip() + ' '
                    
                elif line.startswith('DR   '):
                    # GO 注释 (DR   GO; GO:0001234; ...)
                    if 'go_terms' not in current_record:
                        current_record['go_terms'] = []
                    if 'GO; GO:' in line:
                        parts = line.split(';')
                        if len(parts) >= 2:
                            go = parts[1].strip()
                            if go:
                                current_record['go_terms'].append(go)
                    
                elif line.startswith('CC   -!- CATALYTIC ACTIVITY:'):
                    # EC号在CATALYTIC ACTIVITY注释中
                    if 'ec_numbers' not in current_record:
                        current_record['ec_numbers'] = []
                    current_record['_cc_buffer'] = line[5:].strip()
                    
                elif line.startswith('CC   -!- FUNCTION:'):
                    if 'ec_numbers' not in current_record:
                        current_record['ec_numbers'] = []
                    current_record['_cc_buffer'] = line[5:].strip()
                    
                elif line.startswith('CC       Reaction='):
                    # Reaction行包含EC号，格式：EC 1.2.3.4 或 EC=1.2.3.4
                    if '_cc_buffer' in current_record:
                        reaction_text = line[5:].strip()
                        ec_patterns = re.findall(r'EC\s+([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)', reaction_text, re.IGNORECASE)
                        ec_patterns2 = re.findall(r'EC=([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)', reaction_text, re.IGNORECASE)
                        for ec in ec_patterns + ec_patterns2:
                            if ec not in current_record['ec_numbers']:
                                current_record['ec_numbers'].append(ec)
                                
                elif line.startswith('CC   -!- SUBCELLULAR LOCATION:'):
                    if 'subcellular' not in current_record:
                        current_record['subcellular'] = ''
                    current_record['subcellular'] += line[5:].strip() + ' '
                    current_record['_cc_subcellular_continue'] = True
                    
                elif line.startswith('CC       ') and current_record.get('_cc_subcellular_continue'):
                    current_record['subcellular'] += line[5:].strip() + ' '
                    
                elif line.startswith('CC   -!-'):
                    if '_cc_subcellular_continue' in current_record:
                        del current_record['_cc_subcellular_continue']
                    if '_cc_buffer' in current_record:
                        del current_record['_cc_buffer']
                    
                elif line.startswith('KW   '):
                    if 'keywords' not in current_record:
                        current_record['keywords'] = []
                    kw = line[5:].strip().rstrip(';')
                    if kw:
                        current_record['keywords'].append(kw)
        
        # 最后一个记录
        if current_record:
            records.append(current_record)
        
        # 转换为 DataFrame
        df = pd.DataFrame(records)
        
        # 清理临时字段
        for col in ['_cc_buffer', '_cc_subcellular_continue']:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # 处理列表字段
        for col in ['ec_numbers', 'go_terms', 'keywords']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
        
        return df
    
    def download_and_parse(self, species: str = None) -> pd.DataFrame:
        """下载并解析 SwissProt 数据
        
        Args:
            species: 物种过滤，如 "9606" (human), "83333" (E.coli)
        
        Returns:
            DataFrame with protein data
        """
        dat_file = self.download_sprot()
        print("正在解析数据...")
        df = self.parse_dat_file(dat_file)
        
        if species:
            # 按物种过滤 (简化版)
            df = df[df['organism'].str.contains(species, na=False)]
        
        print(f"解析完成，共 {len(df)} 条记录")
        return df


class MultiTaskDataDownloader:
    """多任务数据下载器"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.uniprot = UniProtDownloader(output_dir)
    
    def download_ec_data(self, min_ec_depth: int = 3) -> pd.DataFrame:
        """下载 EC 注释数据
        
        Args:
            min_ec_depth: EC 号最小深度，3表示至少到第三层如 1.1.1
        """
        print("下载 EC 注释数据...")
        df = self.uniprot.download_and_parse()
        
        # 过滤有 EC 注释的数据
        df_ec = df[df['ec_numbers'].notna() & (df['ec_numbers'] != '')].copy()
        
        # 过滤 EC 深度
        def check_ec_depth(ec_str):
            ecs = ec_str.split(',')
            for ec in ecs:
                parts = ec.strip().split('.')
                if len(parts) >= min_ec_depth:
                    return True
            return False
        
        df_ec = df_ec[df_ec['ec_numbers'].apply(check_ec_depth)]
        df_ec = df_ec[['ID', 'sequence', 'ec_numbers']].copy()
        df_ec.columns = ['id', 'sequence', 'ec_number']
        
        # 清理序列
        df_ec = df_ec[df_ec['sequence'].str.len() > 10]  # 过滤太短的序列
        df_ec = df_ec.dropna()
        
        print(f"EC 数据下载完成: {len(df_ec)} 条")
        return df_ec
    
    def download_localization_data(self) -> pd.DataFrame:
        """下载细胞定位数据"""
        print("下载细胞定位数据...")
        df = self.uniprot.download_and_parse()
        
        # 过滤有定位注释的数据
        df_loc = df[df['subcellular'].notna() & (df['subcellular'] != '')].copy()
        df_loc = df_loc[['ID', 'sequence', 'subcellular']].copy()
        df_loc.columns = ['id', 'sequence', 'location']
        
        # 清理
        df_loc = df_loc[df_loc['sequence'].str.len() > 10]
        df_loc = df_loc.dropna()
        
        print(f"定位数据下载完成: {len(df_loc)} 条")
        return df_loc
    
    def download_function_data(self) -> pd.DataFrame:
        """下载蛋白质功能数据 (基于 Keywords)"""
        print("下载功能注释数据...")
        df = self.uniprot.download_and_parse()
        
        # 过滤有功能注释的数据
        df_func = df[df['keywords'].notna() & (df['keywords'] != '')].copy()
        df_func = df_func[['ID', 'sequence', 'keywords', 'description']].copy()
        df_func.columns = ['id', 'sequence', 'keywords', 'description']
        
        # 过滤短序列
        df_func = df_func[df_func['sequence'].str.len() > 10]
        df_func = df_func.dropna()
        
        print(f"功能数据下载完成: {len(df_func)} 条")
        return df_func
    
    def create_unified_dataset(self) -> pd.DataFrame:
        """创建统一的多任务数据集"""
        print("创建统一数据集...")
        
        # 下载各任务数据
        df_ec = self.download_ec_data()
        df_loc = self.download_localization_data()
        df_func = self.download_function_data()
        
        # 合并数据 (基于 ID)
        df = df_ec.merge(df_loc[['id', 'location']], on='id', how='outer')
        df = df.merge(df_func[['id', 'keywords', 'description']], on='id', how='outer')
        
        # 填充空值
        df['ec_number'] = df['ec_number'].fillna('')
        df['location'] = df['location'].fillna('')
        df['keywords'] = df['keywords'].fillna('')
        df['description'] = df['description'].fillna('')
        
        # 过滤有任意一个标签的数据
        df = df[(df['ec_number'] != '') | (df['location'] != '') | (df['keywords'] != '')]
        
        # 过滤短序列
        df = df[df['sequence'].str.len() > 10]
        df = df.drop_duplicates(subset=['id'])
        
        print(f"统一数据集创建完成: {len(df)} 条记录")
        return df


def main():
    """主函数 - 下载所有数据"""
    downloader = MultiTaskDataDownloader()
    
    print("=" * 60)
    print("开始下载蛋白质数据...")
    print("=" * 60)
    
    # 创建统一数据集
    df = downloader.create_unified_dataset()
    
    # 保存原始数据
    output_file = RAW_DATA_DIR / "protein_data_raw.parquet"
    df.to_parquet(output_file, index=False)
    print(f"\n数据已保存到: {output_file}")
    
    # 显示统计信息
    print("\n数据集统计:")
    print(f"  总记录数: {len(df)}")
    print(f"  有 EC 注释: {(df['ec_number'] != '').sum()}")
    print(f"  有定位注释: {(df['location'] != '').sum()}")
    print(f"  有功能注释: {(df['keywords'] != '').sum()}")


if __name__ == "__main__":
    main()
