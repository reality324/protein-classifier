"""BNN 分类器 - 贝叶斯神经网络 (MC Dropout)"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from scipy.stats import entropy

from .base import ProteinClassifier, ClassifierInfo, register_classifier


class BayesianMLP(nn.Module):
    """带 MC Dropout 的 MLP 网络"""

    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.dropout(features)
        return self.classifier(features)


@register_classifier("bnn")
class BNNClassifier(ProteinClassifier):
    """BNN 贝叶斯神经网络分类器 (MC Dropout)

    原理: 使用 MC Dropout 在测试时进行多次采样，利用预测的方差估计不确定性
    优点: 除了预测类别，还能提供预测的不确定性估计
    缺点: 训练最慢，预测需要多次采样

    Example:
        >>> clf = BNNClassifier(mc_samples=30)
        >>> clf.fit(X_train, y_train)
        >>> y_pred, uncertainty = clf.predict_with_uncertainty(X_test)
    """

    name = "bnn"
    info = ClassifierInfo(
        name="bnn",
        type="pytorch",
        description="BNN - 贝叶斯神经网络 (MC Dropout)，支持不确定性估计",
        supports_uncertainty=True,
        requires_gpu=False,
    )

    def __init__(
        self,
        hidden_dims: list = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        kl_weight: float = 0.1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        mc_samples: int = 30,
        device: str = None,
        random_state: int = 42,
    ):
        """
        Args:
            hidden_dims: 隐藏层维度列表
            dropout: Dropout 比例
            learning_rate: 学习率
            weight_decay: 权重衰减
            kl_weight: KL 散度权重
            batch_size: 批大小
            epochs: 最大训练轮数
            patience: 早停耐心值
            mc_samples: Monte Carlo 采样次数 (用于估计不确定性)
            device: 训练设备
            random_state: 随机种子
        """
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.mc_samples = mc_samples
        self.random_state = random_state

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = None
        self.input_dim_ = None
        self.num_classes_ = None
        self.history_ = {"train_loss": [], "val_loss": [], "val_acc": []}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BNNClassifier":
        """训练 BNN"""
        self.input_dim_ = X.shape[1]
        self.num_classes_ = len(np.unique(y))

        # 构建模型
        self.model = BayesianMLP(
            self.input_dim_,
            self.hidden_dims,
            self.num_classes_,
            self.dropout
        ).to(self.device)

        # 数据
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        train_dataset = TensorDataset(X_tensor, y_tensor)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 优化器 - 训练时启用 Dropout
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 训练 - 始终启用 Dropout
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)

            train_loss /= len(train_dataset)

            # 验证阶段
            val_loss = None
            val_acc = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_x.size(0)

                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                val_loss /= len(val_dataset)
                val_acc = correct / total

                scheduler.step(val_loss)

                self.history_["val_loss"].append(val_loss)
                self.history_["val_acc"].append(val_acc)

                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"[BNNClassifier] 早停于 epoch {epoch + 1}")
                    break

            self.history_["train_loss"].append(train_loss)

            if (epoch + 1) % 10 == 0:
                val_info = f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}" if val_acc else ""
                print(f"[BNNClassifier] Epoch {epoch + 1}/{self.epochs}, train_loss={train_loss:.4f}{val_info}")

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别 (使用 MC Dropout 采样)"""
        _, probs = self._mc_predict(X, self.mc_samples)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率 (使用 MC Dropout 采样)"""
        _, probs = self._mc_predict(X, self.mc_samples)
        return probs

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """带不确定性估计的预测

        Args:
            X: 特征矩阵
            n_samples: MC 采样次数

        Returns:
            (predictions, uncertainties, probs)
            - predictions: 预测类别 (n_samples,)
            - uncertainties: 不确定性分数 (n_samples,) - 预测熵
            - probs: 平均预测概率 (n_samples, n_classes)
        """
        if n_samples is None:
            n_samples = self.mc_samples

        predictions_list = []
        probs_list = []

        # 多次采样
        for _ in range(n_samples):
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)

            predictions_list.append(torch.argmax(outputs, dim=1).cpu().numpy())
            probs_list.append(probs.cpu().numpy())

        predictions_array = np.array(predictions_list)  # (n_samples, n_data)
        probs_array = np.array(probs_list)  # (n_samples, n_data, n_classes)

        # 平均概率
        mean_probs = probs_array.mean(axis=0)
        final_predictions = np.argmax(mean_probs, axis=1)

        # 计算不确定性 (预测熵)
        uncertainties = entropy(mean_probs.T + 1e-10)

        return final_predictions, uncertainties, mean_probs

    def _mc_predict(self, X: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """MC Dropout 预测"""
        predictions = []
        probs = []

        for _ in range(n_samples):
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor)
                prob = torch.softmax(outputs, dim=1)

            predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())
            probs.append(prob.cpu().numpy())

        predictions = np.array(predictions).mean(axis=0)
        probs = np.array(probs).mean(axis=0)

        return predictions.astype(int), probs

    def get_info(self) -> Dict[str, Any]:
        return {
            **super().get_info(),
            "params": {
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "mc_samples": self.mc_samples,
            },
            "input_dim": self.input_dim_,
            "num_classes": self.num_classes_,
            "device": str(self.device),
            "supports_uncertainty": True,
        }

    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim_,
            'num_classes': self.num_classes_,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'mc_samples': self.mc_samples,
            'history': self.history_,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim_ = checkpoint['input_dim']
        self.num_classes_ = checkpoint['num_classes']
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout = checkpoint['dropout']
        self.mc_samples = checkpoint['mc_samples']
        self.history_ = checkpoint.get('history', {})

        self.model = BayesianMLP(
            self.input_dim_,
            self.hidden_dims,
            self.num_classes_,
            self.dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
