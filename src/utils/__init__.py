"""
Utils module
"""
from .metrics import (
    calculate_binary_metrics,
    calculate_multiclass_metrics,
    MetricTracker,
    Evaluator,
)

# Optional visualization module (requires matplotlib)
try:
    from .visualization import (
        TrainingVisualizer,
        ConfusionMatrixPlotter,
        EmbeddingVisualizer,
        create_summary_report,
    )
    _visualization_available = True
except ImportError:
    _visualization_available = False
    TrainingVisualizer = None
    ConfusionMatrixPlotter = None
    EmbeddingVisualizer = None
    create_summary_report = None

__all__ = [
    'calculate_binary_metrics',
    'calculate_multiclass_metrics',
    'MetricTracker',
    'Evaluator',
    'TrainingVisualizer',
    'ConfusionMatrixPlotter',
    'EmbeddingVisualizer',
    'create_summary_report',
]
