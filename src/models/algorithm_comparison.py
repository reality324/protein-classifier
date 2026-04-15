"""
算法对比框架
支持多种算法: Random Forest, XGBoost, SVM, Neural Network, ESM2
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import DATASETS_DIR
from src.data.dataset import ProteinDataset


@dataclass
class AlgorithmConfig:
    """算法配置"""
    name: str
    type: str  # 'traditional' or 'deep_learning'
    params: Dict = field(default_factory=dict)

    def __str__(self):
        return f"{self.name} ({self.type})"


class BaseAlgorithm:
    """算法基类"""

    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        raise NotImplementedError

    def save(self, path: Path):
        """保存模型"""
        raise NotImplementedError

    def load(self, path: Path):
        """加载模型"""
        raise NotImplementedError


# ===================== 传统机器学习算法 =====================

class RandomForestAlgorithm(BaseAlgorithm):
    """随机森林算法"""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self.n_estimators = config.params.get('n_estimators', 100)
        self.max_depth = config.params.get('max_depth', 20)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练随机森林"""
        # 处理多标签
        if len(y.shape) > 1 and y.shape[1] > 1:
            self.model = MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    n_jobs=-1,
                    random_state=42
                )
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=-1,
                random_state=42
            )

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: Path):
        import joblib
        self.model = joblib.load(path)
        self.is_fitted = True


class XGBoostAlgorithm(BaseAlgorithm):
    """XGBoost 算法"""

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            import xgboost as xgb
        except ImportError:
            print("XGBoost 未安装，使用 GradientBoosting 替代")
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.model.fit(X, y)
            self.is_fitted = True
            return

        # 处理多标签
        if len(y.shape) > 1 and y.shape[1] > 1:
            self.model = MultiOutputClassifier(
                xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42
                )
            )
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: Path):
        import joblib
        self.model = joblib.load(path)
        self.is_fitted = True


class SVMAlgorithm(BaseAlgorithm):
    """SVM 算法"""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self.C = config.params.get('C', 1.0)
        self.kernel = config.params.get('kernel', 'rbf')
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 处理多标签
        if len(y.shape) > 1 and y.shape[1] > 1:
            self.model = MultiOutputClassifier(
                SVC(C=self.C, kernel=self.kernel, probability=True, random_state=42)
            )
        else:
            self.model = SVC(C=self.C, kernel=self.kernel, probability=True, random_state=42)

        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: Path):
        import joblib
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load(self, path: Path):
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = True


class LogisticRegressionAlgorithm(BaseAlgorithm):
    """逻辑回归算法"""

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 处理多标签
        if len(y.shape) > 1 and y.shape[1] > 1:
            self.model = MultiOutputClassifier(
                LogisticRegression(max_iter=1000, random_state=42)
            )
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)

        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: Path):
        import joblib
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load(self, path: Path):
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = True


# ===================== 深度学习算法 =====================

class NeuralNetworkAlgorithm(BaseAlgorithm):
    """神经网络算法"""

    def __init__(self, config: AlgorithmConfig, input_dim: int, output_dim: int, task_type: str = 'multilabel'):
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type  # 'multiclass' or 'multilabel'
        self.hidden_dims = config.params.get('hidden_dims', [256, 128])
        self.learning_rate = config.params.get('learning_rate', 0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_model(self) -> nn.Module:
        """构建神经网络"""
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))

        if self.task_type == 'multilabel':
            # 多标签使用 sigmoid
            model = nn.Sequential(*layers)
        else:
            # 多分类使用 log_softmax
            layers.append(nn.LogSoftmax(dim=1))
            model = nn.Sequential(*layers)

        return model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        self.model = self._build_model()

        # 转换为张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 损失函数
        if self.task_type == 'multilabel':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.NLLLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 训练
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)

                if self.task_type == 'multilabel':
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs, batch_y.long())

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        self.is_fitted = True

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)

            if self.task_type == 'multilabel':
                preds = (torch.sigmoid(outputs) > threshold).cpu().numpy()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)

            if self.task_type == 'multilabel':
                probs = torch.sigmoid(outputs).cpu().numpy()
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

        return probs

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path):
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.is_fitted = True


# ===================== 算法工厂 =====================

class AlgorithmFactory:
    """算法工厂"""

    _ALGORITHMS = {
        'random_forest': RandomForestAlgorithm,
        'xgboost': XGBoostAlgorithm,
        'svm': SVMAlgorithm,
        'logistic_regression': LogisticRegressionAlgorithm,
        'neural_network': NeuralNetworkAlgorithm,
    }

    @classmethod
    def create(
        cls,
        name: str,
        input_dim: int = None,
        output_dim: int = None,
        task_type: str = 'multilabel',
        **params
    ) -> BaseAlgorithm:
        """创建算法实例"""
        if name not in cls._ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {name}. Available: {list(cls._ALGORITHMS.keys())}")

        config = AlgorithmConfig(name=name, type='traditional', params=params)
        algorithm_cls = cls._ALGORITHMS[name]

        if name == 'neural_network':
            if input_dim is None or output_dim is None:
                raise ValueError("Neural network requires input_dim and output_dim")
            return NeuralNetworkAlgorithm(config, input_dim, output_dim, task_type)

        return algorithm_cls(config)

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """列出所有可用算法"""
        return list(cls._ALGORITHMS.keys())


# ===================== 算法对比器 =====================

class AlgorithmComparator:
    """算法对比器"""

    def __init__(self):
        self.results = {}
        self.best_algorithm = None
        self.best_score = 0

    def compare(
        self,
        algorithms: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = 'multilabel',
        input_dim: int = None,
        output_dim: int = None,
    ) -> Dict:
        """对比多个算法"""

        results = {}

        for algo_name in algorithms:
            print(f"\n{'='*60}")
            print(f"训练算法: {algo_name}")
            print('='*60)

            try:
                # 创建算法
                if algo_name == 'neural_network':
                    algo = AlgorithmFactory.create(
                        algo_name,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        task_type=task_type,
                    )
                else:
                    algo = AlgorithmFactory.create(algo_name)

                # 训练
                import time
                start_time = time.time()

                if algo_name == 'neural_network':
                    algo.fit(X_train, y_train, epochs=50, batch_size=32)
                else:
                    algo.fit(X_train, y_train)

                train_time = time.time() - start_time

                # 评估
                if task_type == 'multilabel':
                    metrics = self._evaluate_multilabel(algo, X_test, y_test)
                else:
                    metrics = self._evaluate_multiclass(algo, X_test, y_test)

                metrics['train_time'] = train_time

                results[algo_name] = metrics

                # 打印结果
                self._print_results(algo_name, metrics)

                # 更新最佳算法
                current_score = metrics.get('f1_macro', metrics.get('accuracy', 0))
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_algorithm = algo_name

            except Exception as e:
                print(f"算法 {algo_name} 训练失败: {e}")
                results[algo_name] = {'error': str(e)}

        self.results = results
        return results

    def _evaluate_multilabel(
        self,
        algo: BaseAlgorithm,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """评估多标签分类"""
        y_pred = algo.predict(X_test, threshold)
        y_prob = algo.predict_proba(X_test)

        # 二值化评估
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'f1_samples': f1_score(y_test, y_pred, average='samples', zero_division=0),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        }

        # AUC (需要概率)
        try:
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                # 对于多标签多输出
                aucs = []
                for i in range(y_prob.shape[1]):
                    if len(np.unique(y_test[:, i])) > 1:
                        aucs.append(roc_auc_score(y_test[:, i], y_prob[:, i]))
                metrics['auc_macro'] = np.mean(aucs) if aucs else None
        except:
            pass

        return metrics

    def _evaluate_multiclass(
        self,
        algo: BaseAlgorithm,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """评估多分类"""
        y_pred = algo.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        }

        return metrics

    def _print_results(self, algo_name: str, metrics: Dict):
        """打印结果"""
        print(f"\n{algo_name} 结果:")
        for metric_name, value in metrics.items():
            if value is not None and not isinstance(value, str):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")

    def plot_comparison(self, save_path: Path = None):
        """绘制对比图"""
        import matplotlib.pyplot as plt

        # 过滤错误结果
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not valid_results:
            print("没有有效的算法结果可比较")
            return

        # 准备数据
        algorithms = list(valid_results.keys())
        metrics_names = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics_names):
            ax = axes[i]
            values = [valid_results[algo].get(metric, 0) for algo in algorithms]

            bars = ax.bar(algorithms, values, color='steelblue', alpha=0.8)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"对比图已保存: {save_path}")

        return fig

    def print_summary(self):
        """打印总结"""
        print("\n" + "="*60)
        print("算法对比总结")
        print("="*60)

        print(f"\n最佳算法: {self.best_algorithm}")
        print(f"最佳得分: {self.best_score:.4f}")

        print("\n完整结果:")
        for algo, metrics in self.results.items():
            print(f"\n{algo}:")
            if 'error' in metrics:
                print(f"  错误: {metrics['error']}")
            else:
                for k, v in metrics.items():
                    if v is not None and not isinstance(v, str):
                        print(f"  {k}: {v:.4f}")


# ===================== 主函数 =====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='算法对比')
    parser.add_argument('--data', type=str,
                       default=str(DATASETS_DIR / 'train.parquet'),
                       help='数据路径')
    parser.add_argument('--test_data', type=str,
                       default=str(DATASETS_DIR / 'test.parquet'),
                       help='测试数据路径')
    parser.add_argument('--algorithms', '-a', nargs='+',
                       default=['random_forest', 'xgboost', 'neural_network'],
                       help='要对比的算法')
    parser.add_argument('--task', type=str, default='location',
                       choices=['ec', 'location', 'function'],
                       help='任务类型')
    parser.add_argument('--output', '-o', type=str,
                       default='algorithm_comparison.png',
                       help='输出图表路径')

    args = parser.parse_args()

    print("="*60)
    print("蛋白质分类 - 算法对比")
    print("="*60)

    # 加载数据
    from torch.utils.data import DataLoader

    train_dataset = ProteinDataset(args.data)
    test_dataset = ProteinDataset(args.test_data)

    # 准备数据 (简化版)
    X_train = np.array([train_dataset[i]['features'].numpy() for i in range(min(1000, len(train_dataset)))])
    y_train = np.zeros((len(X_train), 30))  # 简化

    X_test = np.array([test_dataset[i]['features'].numpy() for i in range(min(200, len(test_dataset)))])
    y_test = np.zeros((len(X_test), 30))

    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")

    # 对比算法
    comparator = AlgorithmComparator()
    results = comparator.compare(
        algorithms=args.algorithms,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type='multilabel',
        input_dim=X_train.shape[1],
        output_dim=30,
    )

    # 打印总结
    comparator.print_summary()

    # 绘制对比图
    comparator.plot_comparison(Path(args.output))


if __name__ == "__main__":
    main()
