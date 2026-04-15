"""
Data module
"""
from .download import UniProtDownloader, MultiTaskDataDownloader
from .preprocessing import ProteinDataProcessor, preprocess_pipeline, split_dataset
from .featurization import (
    OneHotFeatureExtractor,
    ESMFeatureExtractor,
    ProtBERTFeatureExtractor,
    get_feature_extractor,
)
from .dataset import ProteinDataset, ProteinDatasetWithEmbedding, DataLoaderFactory
from .augmentation import ProteinAugmenter, AUGMENTATION_STRATEGIES
from .location_taxonomy import (
    SUBCELLULAR_LOCATIONS,
    SIMPLIFIED_LOCATIONS,
    LOCATION_MAPPING,
    map_to_simplified_location,
    get_location_keyword_patterns,
    get_all_location_names,
    get_go_mapping,
)

__all__ = [
    # Download
    'UniProtDownloader',
    'MultiTaskDataDownloader',
    # Preprocessing
    'ProteinDataProcessor',
    'preprocess_pipeline',
    'split_dataset',
    # Featurization
    'OneHotFeatureExtractor',
    'ESMFeatureExtractor',
    'ProtBERTFeatureExtractor',
    'get_feature_extractor',
    # Dataset
    'ProteinDataset',
    'ProteinDatasetWithEmbedding',
    'DataLoaderFactory',
    # Augmentation
    'ProteinAugmenter',
    'AUGMENTATION_STRATEGIES',
    # Location taxonomy
    'SUBCELLULAR_LOCATIONS',
    'SIMPLIFIED_LOCATIONS',
    'LOCATION_MAPPING',
    'map_to_simplified_location',
    'get_location_keyword_patterns',
    'get_all_location_names',
    'get_go_mapping',
]
