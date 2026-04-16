#!/usr/bin/env python3
"""
模型评估可视化脚本
用于绘制训练过程曲线、测试指标对比图、多算法效果对比图

支持的功能：
1. 绘制训练/验证损失曲线
2. 绘制各任务（EC、Location、Function）的评估指标
3. 多算法效果对比（Radar Chart、Bar Chart）
4. 混淆矩阵可视化
5. 生成完整的评估报告图

使用方法:
    python scripts/plot_evaluation.py --mode all --log_dir logs/
    python scripts/plot_evaluation.py --mode comparison --results_dir results/
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 设置高质量绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 颜色方案
COLORS = {
    'ec': '#3498db',
    'location': '#e74c3c',
    'function': '#2ecc71',
    'train': '#3498db',
    'val': '#e74c3c',
}

ALGORITHM_COLORS = {
    'random_forest': '#3498db',
    'xgboost': '#2ecc71',
    'svm': '#e74c3c',
    'logistic_regression': '#9b59b6',
    'neural_network': '#f39c12',
    'bnn': '#1abc9c',
}


class EvaluationPlotter:
    """模型评估可视化器"""

    def __init__(self, output_dir: Path = None, figure_size: Tuple[int, int] = (12, 8)):
        self.output_dir = Path(output_dir) if output_dir else Path('./evaluation_plots')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_size = figure_size

    def plot_loss_curves(self, history: List[Dict], save: bool = True) -> plt.Figure:
        """绘制训练损失曲线

        Args:
            history: 训练历史数据列表
            save: 是否保存图片
        """
        df = pd.DataFrame(history)
        epochs = df['epoch'].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 总损失曲线
        ax1 = axes[0]
        ax1.plot(epochs, df['train_loss'], color=COLORS['train'], linewidth=2.5,
                 label='Training Loss', marker='o', markersize=6, alpha=0.9)
        ax1.plot(epochs, df['val_loss'], color=COLORS['val'], linewidth=2.5,
                 label='Validation Loss', marker='s', markersize=6, alpha=0.9)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # 添加最优标记
        best_val_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
        best_val_loss = df['val_loss'].min()
        ax1.axvline(x=best_val_epoch, color='gray', linestyle='--', alpha=0.7)
        ax1.annotate(f'Best: {best_val_loss:.4f}\n(Epoch {int(best_val_epoch)})',
                     xy=(best_val_epoch, best_val_loss),
                     xytext=(best_val_epoch + 0.5, best_val_loss * 1.05),
                     fontsize=10, color='gray',
                     arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

        # 各任务损失曲线
        ax2 = axes[1]
        tasks = [('ec_loss', 'EC Number', COLORS['ec']),
                 ('loc_loss', 'Location', COLORS['location']),
                 ('func_loss', 'Function', COLORS['function'])]

        for task_key, task_name, color in tasks:
            if task_key in df.columns:
                ax2.plot(epochs, df[task_key], color=color, linewidth=2,
                         label=task_name, marker='o', markersize=4, alpha=0.8)

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Task-specific Losses', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'loss_curves.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"损失曲线已保存: {save_path}")

        return fig

    def plot_metrics_comparison(self, test_results: Dict, save: bool = True) -> plt.Figure:
        """绘制测试指标对比图

        Args:
            test_results: 测试结果字典
            save: 是否保存图片
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        tasks = ['ec', 'location', 'function']
        task_names = ['EC Number', 'Cell Location', 'Function']
        task_colors = [COLORS['ec'], COLORS['location'], COLORS['function']]

        for i, (task, name, color) in enumerate(zip(tasks, task_names, task_colors)):
            ax = axes[i]
            results = test_results.get(task, {})

            # 提取指标
            if task == 'location':
                metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            else:
                metrics = ['precision_macro', 'recall_macro', 'f1_macro']

            values = []
            labels = []
            for m in metrics:
                if m in results:
                    values.append(results[m])
                    labels.append(m.replace('_macro', '').replace('_', ' ').title())

            # 绘制水平条形图
            bars = ax.barh(labels, values, color=color, alpha=0.8, edgecolor='white', height=0.6)

            # 添加数值标签
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

            ax.set_xlim(0, max(values) * 1.15)
            ax.set_xlabel('Score', fontsize=12)
            ax.set_title(f'{name} Metrics', fontsize=14, fontweight='bold', color=color)
            ax.grid(True, axis='x', alpha=0.3)
            ax.spines['left'].set_visible(False)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'test_metrics_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"指标对比图已保存: {save_path}")

        return fig

    def plot_training_metrics_over_time(self, history: List[Dict], save: bool = True) -> plt.Figure:
        """绘制训练过程中指标变化趋势

        Args:
            history: 训练历史数据列表
            save: 是否保存图片
        """
        df = pd.DataFrame(history)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # EC 指标
        ec_metrics = ['precision_macro', 'recall_macro', 'f1_macro']
        for i, metric in enumerate(ec_metrics):
            ax = axes[0, i]
            val_data = [h['val_results']['ec'].get(metric, 0) for h in history]
            ax.plot(df['epoch'], val_data, color=COLORS['ec'], linewidth=2,
                    marker='o', markersize=5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'EC - {metric.replace("_", " ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Location 指标
        loc_metrics = ['accuracy', 'f1_macro']
        for i, metric in enumerate(loc_metrics):
            ax = axes[1, i]
            val_data = [h['val_results']['location'].get(metric, 0) for h in history]
            ax.plot(df['epoch'], val_data, color=COLORS['location'], linewidth=2,
                    marker='o', markersize=5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Location - {metric.replace("_", " ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Function 指标
        ax = axes[1, 2]
        val_data = [h['val_results']['function'].get('f1_macro', 0) for h in history]
        ax.plot(df['epoch'], val_data, color=COLORS['function'], linewidth=2,
                marker='o', markersize=5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Function - F1 Score', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'metrics_over_time.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"指标变化趋势图已保存: {save_path}")

        return fig

    def plot_algorithm_comparison(
        self,
        results: Dict[str, Dict],
        task: str = 'location',
        save: bool = True
    ) -> plt.Figure:
        """绘制多算法效果对比图

        Args:
            results: 算法结果字典 {algo_name: {metric: value}}
            task: 任务类型
            save: 是否保存图片
        """
        algorithms = list(results.keys())
        metrics = list(results[algorithms[0]].keys()) if algorithms else []

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 条形图对比
        ax1 = axes[0]
        x = np.arange(len(metrics))
        width = 0.8 / len(algorithms)

        for i, algo in enumerate(algorithms):
            offset = (i - len(algorithms)/2 + 0.5) * width
            values = [results[algo].get(m, 0) for m in metrics]
            bars = ax1.bar(x + offset, values, width,
                          label=algo.replace('_', ' ').title(),
                          color=ALGORITHM_COLORS.get(algo, '#95a5a6'),
                          alpha=0.85, edgecolor='white')

        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title(f'Algorithm Comparison - {task.replace("_", " ").title()}',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=0)
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, axis='y', alpha=0.3)

        # 雷达图
        ax2 = axes[1]
        ax2 = fig.add_subplot(122, polar=True)

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        for algo in algorithms:
            values = [results[algo].get(m, 0) for m in metrics]
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2,
                     label=algo.replace('_', ' ').title(),
                     color=ALGORITHM_COLORS.get(algo, '#95a5a6'))
            ax2.fill(angles, values, alpha=0.15,
                     color=ALGORITHM_COLORS.get(algo, '#95a5a6'))

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=10)
        ax2.set_title(f'Radar Chart - {task.replace("_", " ").title()}',
                      fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f'algorithm_comparison_{task}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"算法对比图已保存: {save_path}")

        return fig

    def plot_all_metrics_heatmap(
        self,
        results: Dict[str, Dict],
        save: bool = True
    ) -> plt.Figure:
        """绘制所有算法和指标的热力图

        Args:
            results: 算法结果字典
            save: 是否保存图片
        """
        algorithms = list(results.keys())
        metrics = list(set(m for algo_results in results.values() for m in algo_results.keys()))

        data = np.zeros((len(algorithms), len(metrics)))
        for i, algo in enumerate(algorithms):
            for j, metric in enumerate(metrics):
                data[i, j] = results[algo].get(metric, 0)

        df = pd.DataFrame(data, index=algorithms, columns=metrics)

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn',
                     center=0.5, vmin=0, vmax=1,
                     linewidths=0.5, ax=ax,
                     cbar_kws={'label': 'Score'})

        ax.set_title('Algorithm Performance Heatmap', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Algorithms', fontsize=12)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'metrics_heatmap.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"热力图已保存: {save_path}")

        return fig

    def plot_comprehensive_report(
        self,
        history: List[Dict],
        test_results: Dict,
        algorithm_results: Optional[Dict[str, Dict]] = None,
        save: bool = True
    ) -> plt.Figure:
        """绘制综合评估报告图

        Args:
            history: 训练历史
            test_results: 测试结果
            algorithm_results: 算法对比结果
            save: 是否保存图片
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

        df = pd.DataFrame(history)
        epochs = df['epoch'].values

        # 1. 损失曲线 (左上 2x2)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(epochs, df['train_loss'], color=COLORS['train'], linewidth=2.5,
                 label='Training Loss', marker='o', markersize=6)
        ax1.plot(epochs, df['val_loss'], color=COLORS['val'], linewidth=2.5,
                 label='Validation Loss', marker='s', markersize=6)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 任务损失 (右上 2x2)
        ax2 = fig.add_subplot(gs[0, 2:])
        tasks = [('ec_loss', 'EC', COLORS['ec']),
                 ('loc_loss', 'Loc', COLORS['location']),
                 ('func_loss', 'Func', COLORS['function'])]
        for task_key, label, color in tasks:
            if task_key in df.columns:
                ax2.plot(epochs, df[task_key], color=color, linewidth=2,
                         label=label, marker='o', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Task-specific Losses', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 测试指标对比 (第二行 1x3)
        tasks_info = [('ec', 'EC Number', COLORS['ec']),
                      ('location', 'Location', COLORS['location']),
                      ('function', 'Function', COLORS['function'])]

        for i, (task, name, color) in enumerate(tasks_info):
            ax = fig.add_subplot(gs[1, i])
            results = test_results.get(task, {})

            if task == 'location':
                metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            else:
                metrics = ['precision_macro', 'recall_macro', 'f1_macro']

            values = []
            labels = []
            for m in metrics:
                if m in results:
                    values.append(results[m])
                    labels.append(m.replace('_macro', '').replace('_', '\n').title())

            bars = ax.barh(labels, values, color=color, alpha=0.85, height=0.6)
            for bar, val in zip(bars, values):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

            ax.set_xlim(0, max(values) * 1.2)
            ax.set_title(f'{name}\nTest Results', fontsize=12, fontweight='bold', color=color)
            ax.grid(True, axis='x', alpha=0.3)
            ax.spines['left'].set_visible(False)

        # 4. 算法对比热力图 (第三行)
        if algorithm_results:
            ax_heatmap = fig.add_subplot(gs[2, :])
            algorithms = list(algorithm_results.keys())
            metrics = list(set(m for r in algorithm_results.values() for m in r.keys()))

            data = np.zeros((len(algorithms), len(metrics)))
            for i, algo in enumerate(algorithms):
                for j, metric in enumerate(metrics):
                    data[i, j] = algorithm_results[algo].get(metric, 0)

            heatmap_df = pd.DataFrame(data, index=[a.replace('_', '\n').title() for a in algorithms],
                                      columns=[m.replace('_', '\n').title() for m in metrics])

            sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn',
                        center=0.5, vmin=0, vmax=1, ax=ax_heatmap,
                        linewidths=0.5, cbar_kws={'label': 'Score'})
            ax_heatmap.set_title('Algorithm Comparison', fontsize=14, fontweight='bold')

        # 添加标题
        fig.suptitle('Protein Classification Model Evaluation Report',
                     fontsize=20, fontweight='bold', y=0.98)

        # 添加生成时间
        fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 ha='right', va='bottom', fontsize=10, color='gray')

        if save:
            save_path = self.output_dir / 'comprehensive_evaluation_report.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"综合评估报告已保存: {save_path}")

        return fig

    def save_summary_table(
        self,
        test_results: Dict,
        algorithm_results: Optional[Dict[str, Dict]] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """生成评估结果汇总表

        Args:
            test_results: 测试结果
            algorithm_results: 算法对比结果
            save: 是否保存CSV
        """
        rows = []

        # 任务结果
        tasks = [('ec', 'EC Number'), ('location', 'Cell Location'), ('function', 'Function')]
        for task, name in tasks:
            results = test_results.get(task, {})
            row = {'Model': 'Multi-task (Current)', 'Task': name}
            for metric, value in results.items():
                if value is not None and isinstance(value, (int, float)):
                    row[metric] = value
            rows.append(row)

        # 算法对比结果
        if algorithm_results:
            for algo, results in algorithm_results.items():
                row = {'Model': algo.replace('_', ' ').title(), 'Task': 'Comparison'}
                for metric, value in results.items():
                    if value is not None and isinstance(value, (int, float)):
                        row[metric] = value
                rows.append(row)

        df = pd.DataFrame(rows)

        if save:
            save_path = self.output_dir / 'evaluation_summary.csv'
            df.to_csv(save_path, index=False)
            print(f"评估汇总表已保存: {save_path}")

        return df


def load_json_results(file_path: Path) -> Dict:
    """加载JSON格式的评估结果"""
    with open(file_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='蛋白质分类模型评估可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 绘制训练损失曲线
  python scripts/plot_evaluation.py --mode loss --log_dir logs/

  # 绘制测试指标
  python scripts/plot_evaluation.py --mode metrics --results logs/training_results.json

  # 绘制综合报告
  python scripts/plot_evaluation.py --mode all --log_dir logs/ --output_dir plots/

  # 绘制算法对比图
  python scripts/plot_evaluation.py --mode comparison --results results/algorithm_results.json
        """
    )

    parser.add_argument('--mode', '-m', type=str, default='all',
                       choices=['loss', 'metrics', 'comparison', 'all', 'report'],
                       help='绘图模式')
    parser.add_argument('--log_dir', type=str, default='logs/',
                       help='训练日志目录')
    parser.add_argument('--results', '-r', type=str, default=None,
                       help='结果JSON文件路径')
    parser.add_argument('--output_dir', '-o', type=str, default='evaluation_plots/',
                       help='输出目录')
    parser.add_argument('--task', type=str, default='location',
                       choices=['ec', 'location', 'function'],
                       help='任务类型（用于算法对比）')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plotter = EvaluationPlotter(output_dir)

    print("=" * 60)
    print("蛋白质分类模型评估可视化")
    print("=" * 60)
    print(f"输出目录: {output_dir}")

    if args.mode in ['loss', 'all']:
        # 加载训练历史
        history_file = Path(args.log_dir) / 'training_results.json'
        if history_file.exists():
            data = load_json_results(history_file)
            history = data.get('history', [])

            if history:
                print("\n绘制损失曲线...")
                plotter.plot_loss_curves(history)

                if args.mode == 'all':
                    print("绘制指标变化趋势...")
                    plotter.plot_training_metrics_over_time(history)
        else:
            print(f"警告: 未找到历史文件 {history_file}")

    if args.mode in ['metrics', 'all']:
        # 加载测试结果
        if args.results:
            results_file = Path(args.results)
        else:
            results_file = Path(args.log_dir) / 'training_results.json'

        if results_file.exists():
            data = load_json_results(results_file)
            test_results = data.get('test_results', {})

            if test_results:
                print("\n绘制测试指标对比图...")
                plotter.plot_metrics_comparison(test_results)

                if args.mode == 'all':
                    print("保存评估汇总表...")
                    plotter.save_summary_table(test_results)
        else:
            print(f"警告: 未找到结果文件 {results_file}")

    if args.mode == 'comparison':
        # 算法对比
        if args.results:
            results_file = Path(args.results)
        else:
            results_file = Path(args.log_dir) / 'algorithm_results.json'

        if results_file.exists():
            algo_results = load_json_results(results_file)

            print("\n绘制算法对比图...")
            plotter.plot_algorithm_comparison(algo_results, args.task)
            plotter.plot_all_metrics_heatmap(algo_results)
        else:
            print(f"警告: 未找到算法结果文件 {results_file}")

    if args.mode == 'report':
        # 综合报告
        history_file = Path(args.log_dir) / 'training_results.json'
        algo_results_file = Path(args.log_dir) / 'algorithm_results.json'

        history = []
        test_results = {}
        algo_results = None

        if history_file.exists():
            data = load_json_results(history_file)
            history = data.get('history', [])
            test_results = data.get('test_results', {})

        if algo_results_file.exists():
            algo_results = load_json_results(algo_results_file)

        if history or test_results:
            print("\n绘制综合评估报告...")
            plotter.plot_comprehensive_report(history, test_results, algo_results)
        else:
            print("错误: 未找到足够的数据来生成报告")

    print("\n" + "=" * 60)
    print(f"所有图片已保存到: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()