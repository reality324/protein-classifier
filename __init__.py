"""
ProteinClassifier - 多任务蛋白质分类器
"""
__version__ = "1.0.0"
__author__ = "ProteinClassifier Team"

from .src.data.download import UniProtDownloader, MultiTaskDataDownloader
from .src.data.preprocessing import ProteinDataProcessor
from .src.models.multi_task_model import MultiTaskProteinClassifier, create_model

__all__ = [
    'UniProtDownloader',
    'MultiTaskDataDownloader',
    'ProteinDataProcessor',
    'MultiTaskProteinClassifier',
    'create_model',
]
