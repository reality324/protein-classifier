"""训练器模块 - 统一的模型训练流程"""
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import time

from ..algorithms import ClassifierRegistry, ProteinClassifier


class Trainer:
    """统一训练器

    负责:
    1. 管理训练流程
    2. 早停和学习率调度
    3. 训练历史记录
    4. 模型保存
    """

    def __init__(
        self,
        model: ProteinClassifier,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            model: 分类器实例
            save_dir: 模型保存目录
            verbose: 是否打印训练信息
        """
        self.model = model
        self.save_dir = Path(save_dir) if save_dir else None
        self.verbose = verbose

        self.best_score: float = 0.0
        self.history: Dict[str, list] = {}
        self.best_epoch: int = 0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """训练模型

        Returns:
            训练结果信息
        """
        start_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"开始训练: {self.model.name}")
            print(f"训练样本: {len(X_train)}, 特征维度: {X_train.shape[1]}")
            if X_val is not None:
                print(f"验证样本: {len(X_val)}")
            print(f"{'='*60}\n")

        # 训练
        self.model.fit(X_train, y_train, X_val, y_val, **kwargs)

        # 计算最终分数
        train_score = self.model.score(X_train, y_train) if hasattr(self.model, 'score') else None
        val_score = None
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            val_score = np.mean(val_preds == y_val)

        elapsed = time.time() - start_time

        # 保存模型
        if self.save_dir and val_score is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.save_dir / f"{self.model.name}_best.pt"
            self.model.save(str(model_path))

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"训练完成!")
            print(f"耗时: {elapsed:.2f}秒")
            if train_score is not None:
                print(f"训练准确率: {train_score:.4f}")
            if val_score is not None:
                print(f"验证准确率: {val_score:.4f}")
            print(f"{'='*60}\n")

        return {
            "model_name": self.model.name,
            "train_time": elapsed,
            "train_accuracy": train_score,
            "val_accuracy": val_score,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.model.predict_proba(X)


class ExperimentRunner:
    """实验运行器 - 批量运行多个编码×算法组合"""

    def __init__(
        self,
        encodings: list = None,
        algorithms: list = None,
        output_dir: str = "results",
        verbose: bool = True,
    ):
        """
        Args:
            encodings: 编码方式列表，如 ['onehot', 'ctd', 'esm2']
            algorithms: 算法列表，如 ['rf', 'xgb', 'mlp']
            output_dir: 结果保存目录
            verbose: 是否打印详细信息
        """
        from ..encodings import EncoderRegistry

        if encodings is None:
            encodings = EncoderRegistry.list_encodings()
        if algorithms is None:
            algorithms = ClassifierRegistry.list_classifiers()

        self.encodings = encodings
        self.algorithms = algorithms
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        self.results: Dict[str, Dict[str, Any]] = {}

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        save_models: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """运行所有编码×算法组合实验

        Returns:
            所有实验结果 {f"{encoding}_{algorithm}": results}
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        results_list = []

        for encoding in self.encodings:
            for algorithm in self.algorithms:
                key = f"{encoding}_{algorithm}"

                if self.verbose:
                    print(f"\n{'#'*60}")
                    print(f"# 运行实验: {key}")
                    print(f"{'#'*60}")

                try:
                    # 创建分类器
                    clf = ClassifierRegistry.get(algorithm)

                    # 训练
                    trainer = Trainer(
                        model=clf,
                        save_dir=self.output_dir / "models" / key if save_models else None,
                        verbose=self.verbose,
                    )
                    train_result = trainer.train(X_train, y_train, X_val, y_val)

                    # 测试集评估
                    test_metrics = {}
                    if X_test is not None and y_test is not None:
                        y_pred = trainer.predict(X_test)
                        y_prob = trainer.predict_proba(X_test)

                        test_metrics = {
                            "test_accuracy": accuracy_score(y_test, y_pred),
                            "test_f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
                            "test_f1_micro": f1_score(y_test, y_pred, average='micro', zero_division=0),
                            "test_precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
                            "test_recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
                        }

                    # 保存结果
                    result = {
                        "encoding": encoding,
                        "algorithm": algorithm,
                        "key": key,
                        "input_dim": X_train.shape[1],
                        **train_result,
                        **test_metrics,
                    }
                    self.results[key] = result
                    results_list.append(result)

                    if self.verbose:
                        print(f"\n结果: {result}")

                except Exception as e:
                    print(f"[ERROR] {key} 训练失败: {e}")
                    self.results[key] = {"error": str(e), "encoding": encoding, "algorithm": algorithm}

        # 保存汇总表格
        if results_list:
            df = pd.DataFrame(results_list)
            output_file = self.output_dir / f"comparison_results.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"所有实验完成! 结果已保存到: {output_file}")
                print(f"{'='*60}\n")

                # 打印汇总
                print("\n实验汇总 (按测试准确率排序):")
                print(df.sort_values("test_accuracy", ascending=False)[
                    ["encoding", "algorithm", "test_accuracy", "test_f1_macro"]
                ].to_string(index=False))

        return self.results
