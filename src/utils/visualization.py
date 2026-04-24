"""可视化工具模块 - 训练曲线、混淆矩阵、特征重要性等"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from pathlib import Path


class TrainingVisualizer:
    """训练过程可视化"""

    def __init__(self, figsize=(12, 4)):
        self.figsize = figsize
        plt.rcParams['font.size'] = 10

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[str] = None,
    ):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Loss 曲线
        if 'train_loss' in history:
            ax = axes[0]
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if 'val_loss' in history:
                ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Accuracy 曲线
        if 'val_acc' in history or 'train_acc' in history:
            ax = axes[1]
            if 'val_acc' in history:
                epochs = range(1, len(history['val_acc']) + 1)
                ax.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存: {save_path}")

        return fig


class ConfusionMatrixPlotter:
    """混淆矩阵可视化"""

    def __init__(self, class_names: Optional[List[str]] = None, figsize=(10, 8)):
        self.class_names = class_names
        self.figsize = figsize

    def plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        save_path: Optional[str] = None,
    ):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap=cmap,
            xticklabels=self.class_names or range(cm.shape[1]),
            yticklabels=self.class_names or range(cm.shape[0]),
            ax=ax,
        )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")

        return fig


class ComparisonVisualizer:
    """实验对比可视化"""

    def __init__(self, figsize=(14, 5)):
        self.figsize = figsize

    def plot_encoding_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "test_accuracy",
        title: str = "Encoding Comparison",
        save_path: Optional[str] = None,
    ):
        """绘制编码方式对比图"""
        # 提取数据
        encodings = list(results.keys())
        values = [results[e].get(metric, 0) for e in encodings]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(encodings, values, color=plt.cm.viridis(np.linspace(0, 1, len(encodings))))

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_heatmap(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Heatmap",
        save_path: Optional[str] = None,
    ):
        """绘制热力图"""
        fig, ax = plt.subplots(figsize=(max(8, len(col_labels)), max(5, len(row_labels))))

        sns.heatmap(
            data,
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Accuracy'},
        )
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def plot_per_class_metrics(
    metrics: Dict[str, Any],
    class_names: List[str],
    save_path: Optional[str] = None,
):
    """绘制每个类别的指标"""
    if 'precision_per_class' not in metrics:
        print("警告: 没有 per-class 指标数据")
        return

    precision = np.array(metrics['precision_per_class'])
    recall = np.array(metrics['recall_per_class'])
    f1 = np.array(metrics['f1_per_class'])

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1', alpha=0.8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
