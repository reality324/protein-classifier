"""MLP 分类器 - 多层感知机"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from .base import ProteinClassifier, ClassifierInfo, register_classifier


class MLP(nn.Module):
    """MLP 网络结构"""

    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@register_classifier("mlp")
class MLPClassifier(ProteinClassifier):
    """MLP 多层感知机分类器

    原理: 由多层神经元组成的前馈神经网络，通过反向传播学习特征表示
    优点: 表达能力强、能学习非线性关系
    缺点: 需要大量数据、训练慢、容易过拟合

    Example:
        >>> clf = MLPClassifier(hidden_dims=[256, 128])
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    name = "mlp"
    info = ClassifierInfo(
        name="mlp",
        type="pytorch",
        description="MLP - 多层感知机，深度神经网络分类器",
        requires_gpu=False,
    )

    def __init__(
        self,
        hidden_dims: list = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        device: str = None,
        random_state: int = 42,
    ):
        """
        Args:
            hidden_dims: 隐藏层维度列表，如 [256, 128]
            dropout: Dropout 比例
            learning_rate: 学习率
            weight_decay: 权重衰减
            batch_size: 批大小
            epochs: 最大训练轮数
            patience: 早停耐心值
            device: 训练设备 ('cuda', 'cpu')
            random_state: 随机种子
        """
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
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
    ) -> "MLPClassifier":
        """训练 MLP"""
        self.input_dim_ = X.shape[1]
        self.num_classes_ = len(np.unique(y))

        # 构建模型
        self.model = MLP(
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

        # 优化器和损失
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 训练
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
                    print(f"[MLPClassifier] 早停于 epoch {epoch + 1}")
                    break

            self.history_["train_loss"].append(train_loss)

            if (epoch + 1) % 10 == 0:
                val_info = f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}" if val_acc else ""
                print(f"[MLPClassifier] Epoch {epoch + 1}/{self.epochs}, train_loss={train_loss:.4f}{val_info}")

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def get_info(self) -> Dict[str, Any]:
        return {
            **super().get_info(),
            "params": {
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
            },
            "input_dim": self.input_dim_,
            "num_classes": self.num_classes_,
            "device": str(self.device),
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
            'history': self.history_,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim_ = checkpoint['input_dim']
        self.num_classes_ = checkpoint['num_classes']
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout = checkpoint['dropout']
        self.history_ = checkpoint.get('history', {})

        self.model = MLP(
            self.input_dim_,
            self.hidden_dims,
            self.num_classes_,
            self.dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
