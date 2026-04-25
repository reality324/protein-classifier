"""多任务学习模块 - 同时预测 EC主类、细胞定位、分子功能"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, Any, List
from pathlib import Path

from .evaluator import Evaluator


class MultiTaskModel(nn.Module):
    """多任务神经网络

    共享编码器 + 三个任务头 (EC主类、细胞定位、分子功能)
    """

    def __init__(
        self,
        input_dim: int,
        task_dims: Dict[str, int],
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: 输入特征维度
            task_dims: 任务名 -> 类别数 的字典
            hidden_dims: 隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()

        self.task_dims = task_dims

        # 共享编码器
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
        self.shared_encoder = nn.Sequential(*layers)

        # 任务头
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_dims.items():
            self.task_heads[task_name] = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor, task: str = None) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (batch, input_dim)
            task: 如果指定，只返回该任务的输出；否则返回所有任务输出

        Returns:
            任务输出
        """
        features = self.shared_encoder(x)

        if task is not None:
            return self.task_heads[task](features)
        else:
            outputs = {name: head(features) for name, head in self.task_heads.items()}
            return outputs


class MultiTaskTrainer:
    """多任务训练器

    支持同时训练 EC + Localization + Function 三个任务
    """

    def __init__(
        self,
        input_dim: int,
        task_dims: Dict[str, int],
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = None,
        random_state: int = 42,
    ):
        """
        Args:
            input_dim: 输入特征维度
            task_dims: 任务配置 {"ec": 6, "localization": 11, "function": 17}
            hidden_dims: 隐藏层维度
            dropout: Dropout 比例
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 训练设备
            random_state: 随机种子
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.task_dims = task_dims
        self.task_weights = {t: 1.0 for t in task_dims}  # 可调整的任务权重

        # 创建模型
        self.model = MultiTaskModel(
            input_dim=input_dim,
            task_dims=task_dims,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.history = {task: {"train_loss": [], "val_loss": [], "val_acc": []}
                       for task in task_dims}
        self.best_state = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train_dict: Dict[str, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        y_val_dict: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        verbose: bool = True,
    ) -> "MultiTaskTrainer":
        """训练多任务模型

        Args:
            X_train: 训练特征
            y_train_dict: 训练标签 {"ec": array, "localization": array, "function": array}
            X_val: 验证特征
            y_val_dict: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            patience: 早停耐心值
            verbose: 是否打印训练信息
        """
        # 转换为张量
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = {t: torch.LongTensor(y).to(self.device) for t, y in y_train_dict.items()}

        train_dataset = TensorDataset(X_train_t, *[y_train_t[t] for t in sorted(y_train_dict.keys())])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 验证集
        if X_val is not None and y_val_dict is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = {t: torch.LongTensor(y).to(self.device) for t, y in y_val_dict.items()}
            val_dataset = TensorDataset(X_val_t, *[y_val_t[t] for t in sorted(y_val_dict.keys())])
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        # 损失函数
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        task_names = sorted(y_train_dict.keys())

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = {t: 0.0 for t in task_names}
            total_samples = 0

            for batch in train_loader:
                X_batch = batch[0]
                y_batch = {task_names[i]: batch[i + 1] for i in range(len(task_names))}

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)

                # 计算每个任务的损失
                loss = 0.0
                for task_name in task_names:
                    task_loss = criterion(outputs[task_name], y_batch[task_name])
                    loss += self.task_weights[task_name] * task_loss
                    train_losses[task_name] += task_loss.item() * X_batch.size(0)

                loss.backward()
                self.optimizer.step()
                total_samples += X_batch.size(0)

            # 记录训练损失
            for task_name in task_names:
                self.history[task_name]["train_loss"].append(
                    train_losses[task_name] / total_samples
                )

            # 验证阶段
            val_losses = {t: 0.0 for t in task_names}
            val_accs = {t: 0.0 for t in task_names}

            if val_loader is not None:
                self.model.eval()
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        X_batch = batch[0]
                        y_batch = {task_names[i]: batch[i + 1] for i in range(len(task_names))}

                        outputs = self.model(X_batch)

                        for task_name in task_names:
                            loss = criterion(outputs[task_name], y_batch[task_name])
                            val_losses[task_name] += loss.item() * X_batch.size(0)

                            preds = outputs[task_name].argmax(dim=1)
                            correct = (preds == y_batch[task_name]).sum().item()
                            val_accs[task_name] += correct

                        val_total += X_batch.size(0)

                # 记录验证指标
                total_val_loss = sum(val_losses.values())
                self.scheduler.step(total_val_loss)

                for task_name in task_names:
                    self.history[task_name]["val_loss"].append(
                        val_losses[task_name] / val_total
                    )
                    self.history[task_name]["val_acc"].append(
                        val_accs[task_name] / val_total
                    )

                # 早停
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    patience_counter = 0
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"[MultiTaskTrainer] 早停于 epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs}"
                if val_loader:
                    msg += f", val_loss={total_val_loss:.4f}"
                for task_name in task_names:
                    if val_loader:
                        msg += f", {task_name}_acc={val_accs[task_name]/val_total:.3f}"
                print(msg)

        # 恢复最佳模型
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self

    def predict(self, X: np.ndarray, task: str) -> np.ndarray:
        """预测单个任务"""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t, task=task)
            predictions = outputs.argmax(dim=1)

        return predictions.cpu().numpy()

    def predict_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """预测所有任务"""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t)

        return {task: out.argmax(dim=1).cpu().numpy() for task, out in outputs.items()}

    def predict_proba(self, X: np.ndarray, task: str) -> np.ndarray:
        """预测单个任务的概率"""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t, task=task)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def evaluate(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """评估所有任务"""
        results = {}
        evaluator = Evaluator()

        for task_name, y_true in y_dict.items():
            y_pred = self.predict(X, task_name)
            y_prob = self.predict_proba(X, task_name)
            metrics = evaluator.evaluate(y_true, y_pred, y_prob)
            results[task_name] = metrics

        return results

    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'task_dims': self.task_dims,
            'history': self.history,
            'task_weights': self.task_weights,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.task_dims = checkpoint['task_dims']
        self.history = checkpoint.get('history', {})
        self.task_weights = checkpoint.get('task_weights', {t: 1.0 for t in self.task_dims})

        # 重建模型
        hidden_dims = [512, 256]  # 默认值
        self.model = MultiTaskModel(
            input_dim=checkpoint['model_state_dict']['shared_encoder.0.weight'].shape[1],
            task_dims=self.task_dims,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
