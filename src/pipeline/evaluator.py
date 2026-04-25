"""评估器模块 - 全面的模型评估工具"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, top_k_accuracy_score,
    cohen_kappa_score, matthews_corrcoef
)


class Evaluator:
    """统一评估器

    提供全面的模型评估指标，包括:
    - 分类指标: Accuracy, Precision, Recall, F1
    - 多标签指标: ROC-AUC, Top-K Accuracy
    - 统计指标: Cohen's Kappa, Matthews Correlation
    - 混淆矩阵和分类报告
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        average: str = "macro",
    ):
        """
        Args:
            class_names: 类别名称列表
            average: 多类别指标的平均方式 ('macro', 'micro', 'weighted')
        """
        self.class_names = class_names
        self.average = average

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """全面评估模型

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率 (可选)

        Returns:
            评估指标字典
        """
        metrics = {}

        # 基本分类指标
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1
        for name, func in [
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ]:
            metrics[f"{name}_{self.average}"] = func(
                y_true, y_pred, average=self.average, zero_division=0
            )
            # Per-class 指标
            per_class = func(y_true, y_pred, average=None, zero_division=0)
            metrics[f"{name}_per_class"] = per_class.tolist()

        # Micro 平均
        metrics["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

        # 加权平均
        metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Top-K Accuracy (跳过如果类别数不匹配)
        if y_prob is not None:
            try:
                n_classes_prob = y_prob.shape[1]
                n_classes_true = len(np.unique(y_true))
                if n_classes_true <= 10 and n_classes_prob >= n_classes_true:
                    for k in [1, 3]:
                        if n_classes_prob >= k:
                            metrics[f"top_{k}_accuracy"] = top_k_accuracy_score(
                                y_true, y_prob, k=k, labels=range(n_classes_true)
                            )
            except Exception:
                pass

        # ROC-AUC (OvR, 多类别)
        if y_prob is not None:
            try:
                n_classes_prob = y_prob.shape[1]
                n_classes_true = len(np.unique(y_true))
                if n_classes_prob >= n_classes_true:
                    if n_classes_true == 2:
                        metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
                    else:
                        metrics["roc_auc_ovr"] = roc_auc_score(
                            y_true, y_prob, multi_class="ovr", average=self.average,
                            labels=range(n_classes_true)
                        )
            except Exception:
                metrics["roc_auc"] = None

        # 统计指标
        metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
        metrics["matthews_corr"] = matthews_corrcoef(y_true, y_pred)

        # 混淆矩阵
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        # 分类报告
        metrics["classification_report"] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )

        return metrics

    def evaluate_model(
        self,
        model,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        """评估模型性能

        Args:
            model: 分类器 (需要有 predict 和 predict_proba 方法)
            X: 特征矩阵
            y_true: 真实标签

        Returns:
            评估结果
        """
        y_pred = model.predict(X)

        try:
            y_prob = model.predict_proba(X)
        except Exception:
            y_prob = None

        return self.evaluate(y_true, y_pred, y_prob)

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """对比多个模型

        Args:
            models: {name: model} 字典
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            {name: metrics} 字典
        """
        results = {}
        for name, model in models.items():
            try:
                results[name] = self.evaluate_model(model, X_test, y_test)
            except Exception as e:
                results[name] = {"error": str(e)}

        return results

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str,
    ):
        """保存评估结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 分离可序列化和不可序列化的部分
        serializable = {}
        for k, v in results.items():
            if isinstance(v, (dict, list, str, float, int, bool, type(None))):
                serializable[k] = v

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        print(f"评估结果已保存到: {output_path}")

    def print_summary(self, metrics: Dict[str, Any]):
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print("模型评估摘要")
        print("=" * 60)

        key_metrics = [
            ("Accuracy", "accuracy"),
            ("F1 (Macro)", "f1_macro"),
            ("F1 (Micro)", "f1_micro"),
            ("Precision (Macro)", "precision_macro"),
            ("Recall (Macro)", "recall_macro"),
            ("ROC-AUC", "roc_auc"),
            ("Cohen's Kappa", "cohen_kappa"),
            ("Matthews Corr", "matthews_corr"),
        ]

        for name, key in key_metrics:
            if key in metrics and metrics[key] is not None:
                if isinstance(metrics[key], (int, float)):
                    print(f"  {name:20s}: {metrics[key]:.4f}")

        print("=" * 60)
