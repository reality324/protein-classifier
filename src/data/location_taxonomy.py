"""
细胞定位分类定义
基于 UniProt 和 GO Subcellular Location Ontology
"""
from typing import Dict, List, Tuple


# 标准亚细胞定位分类
SUBCELLULAR_LOCATIONS = {
    # ========== 细胞核 ==========
    'Nucleus': {
        'keywords': ['nucleus', 'nuclear', 'nucleolus', 'chromosome', 'chromatin'],
        'go_id': 'GO:0005634',
    },
    'Nucleolus': {
        'keywords': ['nucleolus', 'nucleolar'],
        'go_id': 'GO:0005730',
    },
    'Nuclear membrane': {
        'keywords': ['nuclear membrane', 'nuclear envelope', 'nuclear inner membrane'],
        'go_id': 'GO:0031965',
    },

    # ========== 细胞质 ==========
    'Cytoplasm': {
        'keywords': ['cytoplasm', 'cytosol', 'cytoplasmatic', 'cytoplasmic'],
        'go_id': 'GO:0005737',
    },
    'Cytoskeleton': {
        'keywords': ['cytoskeleton', 'actin', 'microtubule', 'intermediate filament'],
        'go_id': 'GO:0005856',
    },

    # ========== 线粒体 ==========
    'Mitochondrion': {
        'keywords': ['mitochondrion', 'mitochondrial', 'mitochondria'],
        'go_id': 'GO:0005739',
    },
    'Mitochondrial matrix': {
        'keywords': ['mitochondrial matrix'],
        'go_id': 'GO:0005759',
    },
    'Mitochondrial membrane': {
        'keywords': ['mitochondrial inner membrane', 'mitochondrial outer membrane'],
        'go_id': 'GO:0031966',
    },

    # ========== 内质网/高尔基 ==========
    'Endoplasmic reticulum': {
        'keywords': ['endoplasmic reticulum', 'er membrane', 'er lumen'],
        'go_id': 'GO:0005783',
    },
    'Rough ER': {
        'keywords': ['rough endoplasmic reticulum', 'rough er'],
        'go_id': 'GO:0005791',
    },
    'Smooth ER': {
        'keywords': ['smooth endoplasmic reticulum', 'smooth er'],
        'go_id': 'GO:0005792',
    },
    'Golgi apparatus': {
        'keywords': ['golgi', 'golgi apparatus', 'golgi body', 'golgi stack'],
        'go_id': 'GO:0005794',
    },

    # ========== 膜结构 ==========
    'Cell membrane': {
        'keywords': ['plasma membrane', 'cell membrane', 'cell surface'],
        'go_id': 'GO:0005886',
    },
    'Membrane': {
        'keywords': ['membrane'],
        'go_id': 'GO:0016020',
    },
    'Membrane raft': {
        'keywords': ['membrane raft', 'lipid raft'],
        'go_id': 'GO:0042710',
    },

    # ========== 囊泡 ==========
    'Lysosome': {
        'keywords': ['lysosome', 'lysosomal', 'lysozyme'],
        'go_id': 'GO:0005764',
    },
    'Peroxisome': {
        'keywords': ['peroxisome', 'peroxisomal', 'microbody'],
        'go_id': 'GO:0005777',
    },
    'Endosome': {
        'keywords': ['endosome', 'endosomal', 'early endosome', 'late endosome'],
        'go_id': 'GO:0005768',
    },
    'Secretory granule': {
        'keywords': ['secretory granule', 'secretory vesicle', 'granule'],
        'go_id': 'GO:0030141',
    },

    # ========== 其他 ==========
    'Ribosome': {
        'keywords': ['ribosome', 'ribosomal', 'ribosome subunit'],
        'go_id': 'GO:0005840',
    },
    'Cytoplasmic vesicle': {
        'keywords': ['cytoplasmic vesicle', 'transport vesicle', 'synaptic vesicle'],
        'go_id': 'GO:0030135',
    },
    'Cytoskeleton': {
        'keywords': ['cytoskeleton', 'actin filament', 'microtubule'],
        'go_id': 'GO:0005856',
    },
    'Centrosome': {
        'keywords': ['centrosome', 'centriole', 'centrosomal'],
        'go_id': 'GO:0005813',
    },
    'Vacuole': {
        'keywords': ['vacuole', 'vacuolar'],
        'go_id': 'GO:0005773',
    },
    'Chloroplast': {
        'keywords': ['chloroplast', 'chloroplastic'],
        'go_id': 'GO:0009507',
    },

    # ========== 分泌/细胞外 ==========
    'Secreted': {
        'keywords': ['secreted', 'secretory', 'extracellular', 'extracellular space',
                     'extracellular region'],
        'go_id': 'GO:0005576',
    },
    'Extracellular space': {
        'keywords': ['extracellular space', 'extracellular region'],
        'go_id': 'GO:0005615',
    },
    'Cell wall': {
        'keywords': ['cell wall', 'cell wall fraction'],
        'go_id': 'GO:0005618',
    },
    'Cell junction': {
        'keywords': ['cell junction', 'synapse', 'synaptic', 'gap junction',
                     'adherens junction', 'tight junction'],
        'go_id': 'GO:0005911',
    },

    # ========== 未知 ==========
    'Unknown': {
        'keywords': ['unknown', 'not localized'],
        'go_id': None,
    },
}


# 简化版定位分类 (用于快速训练)
SIMPLIFIED_LOCATIONS = [
    'Nucleus',
    'Cytoplasm',
    'Mitochondrion',
    'Cell membrane',
    'Endoplasmic reticulum',
    'Golgi apparatus',
    'Lysosome',
    'Peroxisome',
    'Ribosome',
    'Secreted',
    'Cytoskeleton',
    'Endosome',
    'Cytoplasmic vesicle',
    'Vacuole',
    'Unknown',
]


# 定位分类到简化分类的映射
LOCATION_MAPPING = {
    'Nucleus': 'Nucleus',
    'Nucleolus': 'Nucleus',
    'Nuclear membrane': 'Nucleus',
    'Cytoplasm': 'Cytoplasm',
    'Cytoskeleton': 'Cytoskeleton',
    'Cytoplasmic vesicle': 'Cytoplasmic vesicle',
    'Mitochondrion': 'Mitochondrion',
    'Mitochondrial matrix': 'Mitochondrion',
    'Mitochondrial membrane': 'Mitochondrion',
    'Cell membrane': 'Cell membrane',
    'Membrane': 'Cell membrane',
    'Membrane raft': 'Cell membrane',
    'Endoplasmic reticulum': 'Endoplasmic reticulum',
    'Rough ER': 'Endoplasmic reticulum',
    'Smooth ER': 'Endoplasmic reticulum',
    'Golgi apparatus': 'Golgi apparatus',
    'Lysosome': 'Lysosome',
    'Peroxisome': 'Peroxisome',
    'Endosome': 'Endosome',
    'Secretory granule': 'Secreted',
    'Ribosome': 'Ribosome',
    'Secreted': 'Secreted',
    'Extracellular space': 'Secreted',
    'Cell wall': 'Secreted',
    'Cell junction': 'Cell membrane',
    'Centrosome': 'Cytoplasm',
    'Vacuole': 'Vacuole',
    'Chloroplast': 'Cytoplasm',
    'Unknown': 'Unknown',
}


def get_location_keyword_patterns() -> Dict[str, List[str]]:
    """获取定位关键词模式"""
    patterns = {}
    for loc_name, loc_info in SUBCELLULAR_LOCATIONS.items():
        patterns[loc_name] = loc_info.get('keywords', [])
    return patterns


def map_to_simplified_location(location: str) -> str:
    """将定位映射到简化分类"""
    location_lower = location.lower()

    for loc_name, loc_info in SUBCELLULAR_LOCATIONS.items():
        keywords = loc_info.get('keywords', [])
        for kw in keywords:
            if kw in location_lower:
                return LOCATION_MAPPING.get(loc_name, loc_name)

    return 'Unknown'


def get_all_location_names() -> List[str]:
    """获取所有定位名称"""
    return list(SUBCELLULAR_LOCATIONS.keys())


def get_go_mapping() -> Dict[str, str]:
    """获取 GO 映射"""
    mapping = {}
    for loc_name, loc_info in SUBCELLULAR_LOCATIONS.items():
        if loc_info.get('go_id'):
            mapping[loc_name] = loc_info['go_id']
    return mapping


if __name__ == "__main__":
    print("细胞定位分类:")
    print(f"总分类数: {len(SUBCELLULAR_LOCATIONS)}")
    print(f"简化分类数: {len(SIMPLIFIED_LOCATIONS)}")

    print("\n简化分类:")
    for loc in SIMPLIFIED_LOCATIONS:
        print(f"  - {loc}")

    print("\n测试映射:")
    test_locs = [
        "Nucleus, Chromosome",
        "Cytoplasm, Cytoskeleton",
        "Mitochondrion inner membrane",
        "Secreted, Extracellular",
    ]
    for loc in test_locs:
        print(f"  '{loc}' -> '{map_to_simplified_location(loc)}'")
