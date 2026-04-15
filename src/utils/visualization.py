"""
可视化工具 - 训练过程可视化、结果分析
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """训练过程可视化"""
    
    def __init__(self, log_dir: Path, output_dir: Path = None):
        self.log_dir = Path(log_dir)
        self.output_dir = output_dir or self.log_dir / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = None
        self.results = None
    
    def load_training_history(self, history_file: str = 'training_results.json'):
        """加载训练历史"""
        history_path = self.log_dir / history_file
        
        if not history_path.exists():
            print(f"警告: 历史文件不存在 {history_path}")
            return
        
        with open(history_path, 'r') as f:
            data = json.load(f)
            self.results = data
            self.history = pd.DataFrame(data['history'])
    
    def plot_loss_curves(self, save: bool = True) -> plt.Figure:
        """绘制损失曲线"""
        if self.history is None:
            print("请先加载训练历史")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = self.history['epoch']
        
        # 训练损失
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        
        # 验证损失
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
            print(f"损失曲线已保存: {self.output_dir / 'loss_curves.png'}")
        
        return fig
    
    def plot_task_losses(self, save: bool = True) -> plt.Figure:
        """绘制各任务损失"""
        if self.history is None:
            print("请先加载训练历史")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        tasks = ['ec_loss', 'loc_loss', 'func_loss']
        titles = ['EC Number Loss', 'Location Loss', 'Function Loss']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (task, title, color) in enumerate(zip(tasks, titles, colors)):
            ax = axes[i]
            ax.plot(self.history['epoch'], self.history[task], color=color, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'task_losses.png', dpi=150, bbox_inches='tight')
            print(f"任务损失已保存: {self.output_dir / 'task_losses.png'}")
        
        return fig
    
    def plot_metrics(self, save: bool = True) -> plt.Figure:
        """绘制评估指标"""
        if self.results is None or 'test_results' not in self.results:
            print("没有测试结果数据")
            return None
        
        test_results = self.results['test_results']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # EC 指标
        ax1 = axes[0]
        ec_metrics = {
            'Precision': test_results['ec']['precision_macro'],
            'Recall': test_results['ec']['recall_macro'],
            'F1': test_results['ec']['f1_macro'],
        }
        bars1 = ax1.bar(ec_metrics.keys(), ec_metrics.values(), color='#1f77b4')
        ax1.set_ylim(0, 1)
        ax1.set_title('EC Number Classification')
        ax1.set_ylabel('Score')
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Location 指标
        ax2 = axes[1]
        loc_metrics = {
            'Accuracy': test_results['location']['accuracy'],
            'Precision': test_results['location']['precision_macro'],
            'Recall': test_results['location']['recall_macro'],
            'F1': test_results['location']['f1_macro'],
        }
        bars2 = ax2.bar(loc_metrics.keys(), loc_metrics.values(), color='#ff7f0e')
        ax2.set_ylim(0, 1)
        ax2.set_title('Location Classification')
        ax2.set_ylabel('Score')
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Function 指标
        ax3 = axes[2]
        func_metrics = {
            'Precision': test_results['function']['precision_macro'],
            'Recall': test_results['function']['recall_macro'],
            'F1': test_results['function']['f1_macro'],
        }
        bars3 = ax3.bar(func_metrics.keys(), func_metrics.values(), color='#2ca02c')
        ax3.set_ylim(0, 1)
        ax3.set_title('Function Classification')
        ax3.set_ylabel('Score')
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'test_metrics.png', dpi=150, bbox_inches='tight')
            print(f"测试指标已保存: {self.output_dir / 'test_metrics.png'}")
        
        return fig
    
    def plot_all(self):
        """绘制所有图表"""
        self.plot_loss_curves()
        self.plot_task_losses()
        self.plot_metrics()
        print(f"\n所有图表已保存到: {self.output_dir}")


class ConfusionMatrixPlotter:
    """混淆矩阵可视化"""
    
    def __init__(self, class_names: List[str], output_dir: Path = None):
        self.class_names = class_names
        self.output_dir = output_dir or Path('.')
    
    def plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Confusion Matrix',
        save_path: Path = None,
        normalize: bool = True,
    ) -> plt.Figure:
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names if len(self.class_names) <= 15 else 'auto',
            yticklabels=self.class_names if len(self.class_names) <= 15 else 'auto',
            ax=ax,
            cbar_kws={'shrink': 0.8}
        )
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        # 旋转标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")
        
        return fig


class EmbeddingVisualizer:
    """嵌入可视化 (使用 PCA/t-SNE)"""
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
    
    def reduce_dimension(
        self,
        embeddings: np.ndarray,
        method: str = 'pca'
    ) -> np.ndarray:
        """降维"""
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=self.n_components)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=self.n_components)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return reducer.fit_transform(embeddings)
    
    def plot_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: List[str],
        title: str = 'Embedding Visualization',
        method: str = 'pca',
        save_path: Path = None,
    ) -> plt.Figure:
        """绘制嵌入可视化"""
        # 降维
        coords = self.reduce_dimension(embeddings, method)
        
        # 创建 DataFrame
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'label': labels,
            'name': [label_names[l] for l in labels]
        })
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 10))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = df['label'] == label
            ax.scatter(
                df.loc[mask, 'x'],
                df.loc[mask, 'y'],
                c=[colors[i]],
                label=label_names[label] if label < len(label_names) else f'Class {label}',
                alpha=0.6,
                s=20
            )
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"嵌入可视化已保存: {save_path}")
        
        return fig


def create_summary_report(
    results: Dict,
    output_path: Path,
    training_config: Dict = None,
):
    """生成训练报告"""
    
    report = []
    report.append("# 蛋白质分类器训练报告\n")
    report.append(f"生成时间: {pd.Timestamp.now()}\n")
    
    # 训练配置
    if training_config:
        report.append("## 训练配置\n")
        for key, value in training_config.items():
            report.append(f"- {key}: {value}")
        report.append("\n")
    
    # 测试结果
    if 'test_results' in results:
        report.append("## 测试结果\n")
        
        test_results = results['test_results']
        
        # EC Number
        if 'ec' in test_results:
            report.append("### EC Number 分类\n")
            ec = test_results['ec']
            report.append(f"- Precision (Macro): {ec['precision_macro']:.4f}")
            report.append(f"- Recall (Macro): {ec['recall_macro']:.4f}")
            report.append(f"- F1 (Macro): {ec['f1_macro']:.4f}")
            if ec.get('auc_macro'):
                report.append(f"- AUC (Macro): {ec['auc_macro']:.4f}")
            report.append("\n")
        
        # Location
        if 'location' in test_results:
            report.append("### 细胞定位分类\n")
            loc = test_results['location']
            report.append(f"- Accuracy: {loc['accuracy']:.4f}")
            report.append(f"- Precision (Macro): {loc['precision_macro']:.4f}")
            report.append(f"- Recall (Macro): {loc['recall_macro']:.4f}")
            report.append(f"- F1 (Macro): {loc['f1_macro']:.4f}")
            report.append("\n")
        
        # Function
        if 'function' in test_results:
            report.append("### 蛋白质功能分类\n")
            func = test_results['function']
            report.append(f"- Precision (Macro): {func['precision_macro']:.4f}")
            report.append(f"- Recall (Macro): {func['recall_macro']:.4f}")
            report.append(f"- F1 (Macro): {func['f1_macro']:.4f}")
            if func.get('auc_macro'):
                report.append(f"- AUC (Macro): {func['auc_macro']:.4f}")
            report.append("\n")
    
    # 保存报告
    with open(output_path, 'w') as f:
        f.writelines('\n'.join(report))
    
    print(f"报告已保存: {output_path}")


if __name__ == "__main__":
    # 示例用法
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    visualizer = TrainingVisualizer(
        Path(args.log_dir),
        Path(args.output_dir) if args.output_dir else None
    )
    
    visualizer.load_training_history()
    visualizer.plot_all()
