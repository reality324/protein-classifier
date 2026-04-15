"""
评估指标模块
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
import warnings
warnings.filterwarnings('ignore')


def calculate_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """计算二分类/多标签分类指标
    
    Args:
        y_true: 真实标签 (n_samples, n_classes)
        y_pred: 预测标签 (n_samples, n_classes)
        y_prob: 预测概率 (用于 AUC)
        threshold: 预测阈值
    
    Returns:
        指标字典
    """
    # 预测二值化
    if y_pred.ndim == 2:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_true.flatten(), y_pred_binary.flatten()),
        'precision_macro': precision_score(y_true, y_pred_binary, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred_binary, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred_binary, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred_binary, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred_binary, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred_binary, average='micro', zero_division=0),
    }
    
    # AUC
    if y_prob is not None and y_true.sum() > 0:
        try:
            metrics['auc_macro'] = roc_auc_score(y_true, y_prob, average='macro')
            metrics['auc_micro'] = roc_auc_score(y_true, y_prob, average='micro')
        except ValueError:
            metrics['auc_macro'] = None
            metrics['auc_micro'] = None
    else:
        metrics['auc_macro'] = None
        metrics['auc_micro'] = None
    
    return metrics


def calculate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, any]:
    """计算多分类指标
    
    Args:
        y_true: 真实标签 (n_samples,)
        y_pred: 预测标签 (n_samples,)
        y_prob: 预测概率 (n_samples, n_classes)
        class_names: 类别名称
    
    Returns:
        指标字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Top-k 准确率
    if y_prob is not None:
        for k in [1, 3, 5]:
            if y_prob.shape[1] >= k:
                metrics[f'top{k}_accuracy'] = top_k_accuracy_score(y_true, y_prob, k=k)
    
    # Top-k 准确率
    if y_prob is not None:
        try:
            metrics['auc_macro'] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
        except ValueError:
            metrics['auc_macro'] = None
    
    return metrics


class MetricTracker:
    """指标跟踪器 (用于训练过程)"""
    
    def __init__(self, tasks: List[str] = ['ec', 'loc', 'func']):
        self.tasks = tasks
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.losses = {task: [] for task in self.tasks}
        self.losses['total'] = []
        self.metrics = {}
    
    def update(self, losses: Dict[str, float], metrics: Optional[Dict[str, float]] = None):
        """更新指标"""
        if 'total' in losses:
            self.losses['total'].append(losses['total'])
        
        for task in self.tasks:
            if task in losses:
                self.losses[task].append(losses[task])
        
        if metrics:
            for key, value in metrics.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
    
    def get_average(self) -> Dict[str, float]:
        """获取平均指标"""
        result = {}
        
        for key, values in self.losses.items():
            if len(values) > 0:
                result[f'{key}_loss'] = np.mean(values)
        
        for key, values in self.metrics.items():
            if len(values) > 0:
                result[key] = np.mean(values)
        
        return result
    
    def get_latest(self) -> Dict[str, float]:
        """获取最新指标"""
        result = {}
        
        for key, values in self.losses.items():
            if len(values) > 0:
                result[f'{key}_loss'] = values[-1]
        
        for key, values in self.metrics.items():
            if len(values) > 0:
                result[key] = values[-1]
        
        return result


class Evaluator:
    """评估器"""
    
    def __init__(
        self,
        ec_classes: int,
        loc_classes: int,
        func_classes: int,
        ec_threshold: float = 0.5,
    ):
        self.ec_classes = ec_classes
        self.loc_classes = loc_classes
        self.func_classes = func_classes
        self.ec_threshold = ec_threshold
        
        # 存储预测和标签
        self.reset()
    
    def reset(self):
        """重置"""
        self.ec_preds = []
        self.ec_targets = []
        self.loc_preds = []
        self.loc_targets = []
        self.func_preds = []
        self.func_targets = []
    
    def update(
        self,
        ec_outputs: torch.Tensor,
        loc_outputs: torch.Tensor,
        func_outputs: torch.Tensor,
        ec_targets: torch.Tensor,
        loc_targets: torch.Tensor,
        func_targets: torch.Tensor,
    ):
        """更新预测和标签"""
        # EC (多标签)
        ec_probs = torch.sigmoid(ec_outputs).cpu().numpy()
        ec_pred = (ec_probs >= self.ec_threshold).astype(int)
        
        self.ec_preds.append(ec_pred)
        self.ec_targets.append(ec_targets.cpu().numpy())
        
        # Location (多分类)
        loc_pred = loc_outputs.argmax(dim=1).cpu().numpy()
        self.loc_preds.append(loc_pred)
        self.loc_targets.append(loc_targets.cpu().numpy())
        
        # Function (多标签)
        func_probs = torch.sigmoid(func_outputs).cpu().numpy()
        func_pred = (func_probs >= self.ec_threshold).astype(int)
        
        self.func_preds.append(func_pred)
        self.func_targets.append(func_targets.cpu().numpy())
    
    def compute(self) -> Dict[str, any]:
        """计算所有指标"""
        results = {}
        
        # EC 指标
        ec_preds = np.vstack(self.ec_preds)
        ec_targets = np.vstack(self.ec_targets)
        results['ec'] = calculate_binary_metrics(
            ec_targets, ec_preds, y_prob=ec_preds
        )
        
        # Location 指标
        loc_preds = np.concatenate(self.loc_preds)
        loc_targets = np.concatenate(self.loc_targets)
        results['location'] = calculate_multiclass_metrics(
            loc_targets, loc_preds
        )
        
        # Function 指标
        func_preds = np.vstack(self.func_preds)
        func_targets = np.vstack(self.func_targets)
        results['function'] = calculate_binary_metrics(
            func_targets, func_preds, y_prob=func_preds
        )
        
        return results
    
    def print_summary(self, results: Dict[str, any]):
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print("评估结果摘要")
        print("=" * 60)
        
        # EC 指标
        print("\n📊 EC Number 预测 (多标签):")
        print(f"  Precision (Macro): {results['ec']['precision_macro']:.4f}")
        print(f"  Recall (Macro): {results['ec']['recall_macro']:.4f}")
        print(f"  F1 (Macro): {results['ec']['f1_macro']:.4f}")
        if results['ec']['auc_macro']:
            print(f"  AUC (Macro): {results['ec']['auc_macro']:.4f}")
        
        # Location 指标
        print("\n📊 细胞定位预测 (多分类):")
        print(f"  Accuracy: {results['location']['accuracy']:.4f}")
        print(f"  Precision (Macro): {results['location']['precision_macro']:.4f}")
        print(f"  Recall (Macro): {results['location']['recall_macro']:.4f}")
        print(f"  F1 (Macro): {results['location']['f1_macro']:.4f}")
        if 'top1_accuracy' in results['location']:
            print(f"  Top-1 Accuracy: {results['location']['top1_accuracy']:.4f}")
        if 'top3_accuracy' in results['location']:
            print(f"  Top-3 Accuracy: {results['location']['top3_accuracy']:.4f}")
        
        # Function 指标
        print("\n📊 蛋白质功能预测 (多标签):")
        print(f"  Precision (Macro): {results['function']['precision_macro']:.4f}")
        print(f"  Recall (Macro): {results['function']['recall_macro']:.4f}")
        print(f"  F1 (Macro): {results['function']['f1_macro']:.4f}")
        if results['function']['auc_macro']:
            print(f"  AUC (Macro): {results['function']['auc_macro']:.4f}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    # 测试指标计算
    # 多标签测试
    y_true_ml = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    y_pred_ml = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
    
    metrics = calculate_binary_metrics(y_true_ml, y_pred_ml)
    print("多标签分类指标:")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")
    
    # 多分类测试
    y_true_mc = np.array([0, 1, 2, 1, 0, 2])
    y_pred_mc = np.array([0, 1, 1, 1, 0, 2])
    y_prob_mc = np.random.rand(6, 3)
    y_prob_mc = y_prob_mc / y_prob_mc.sum(axis=1, keepdims=True)
    
    metrics = calculate_multiclass_metrics(y_true_mc, y_pred_mc, y_prob_mc)
    print("\n多分类指标:")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")
