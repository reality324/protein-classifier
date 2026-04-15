"""
多任务蛋白质分类模型
基于共享特征表示的多任务学习框架
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SharedFeatureExtractor(nn.Module):
    """共享特征提取层"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            self.layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.output_dim = prev_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TaskSpecificHead(nn.Module):
    """任务特定输出头"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
        task_type: str = 'multi_class'  # 'multi_class' or 'multi_label'
    ):
        super().__init__()
        
        self.task_type = task_type
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.output = nn.Linear(hidden_dim, num_classes)
        
        if task_type == 'multi_label':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None  # CrossEntropy 会处理
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared(x)
        x = self.output(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        return x


class MultiTaskProteinClassifier(nn.Module):
    """多任务蛋白质分类器
    
    统一模型同时预测:
    1. EC Number (多标签分类)
    2. 蛋白质功能 (多标签分类)
    3. 细胞定位 (多分类)
    """
    
    def __init__(
        self,
        input_dim: int,
        ec_num_classes: int,
        loc_num_classes: int,
        func_num_classes: int,
        shared_hidden_dims: List[int] = [512, 256],
        task_hidden_dims: List[int] = [128],
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # 共享特征提取器
        self.shared_extractor = SharedFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=shared_hidden_dims,
            dropout=dropout,
        )
        shared_output_dim = self.shared_extractor.output_dim
        
        # 任务特定头
        self.ec_head = TaskSpecificHead(
            input_dim=shared_output_dim,
            num_classes=ec_num_classes,
            hidden_dim=task_hidden_dims[0],
            dropout=dropout,
            task_type='multi_label',
        )
        
        self.loc_head = TaskSpecificHead(
            input_dim=shared_output_dim,
            num_classes=loc_num_classes,
            hidden_dim=task_hidden_dims[0],
            dropout=dropout,
            task_type='multi_class',
        )
        
        self.func_head = TaskSpecificHead(
            input_dim=shared_output_dim,
            num_classes=func_num_classes,
            hidden_dim=task_hidden_dims[0],
            dropout=dropout,
            task_type='multi_label',
        )
    
    def forward(
        self, 
        features: torch.Tensor,
        return_shared: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 输入特征 (batch_size, input_dim)
            return_shared: 是否返回共享特征
        
        Returns:
            dict with keys: 'ec', 'loc', 'func'
        """
        # 共享特征
        shared_features = self.shared_extractor(features)
        
        # 任务特定输出
        outputs = {
            'ec': self.ec_head(shared_features),      # Sigmoid 激活
            'loc': self.loc_head(shared_features),     # Logits (用于 CrossEntropy)
            'func': self.func_head(shared_features),  # Sigmoid 激活
        }
        
        if return_shared:
            outputs['shared'] = shared_features
        
        return outputs


class MultiTaskLoss(nn.Module):
    """多任务损失函数
    
    支持:
    - 类别权重调整
    - 任务权重调整
    - 不平衡标签处理
    """
    
    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        use_pos_weight: bool = False,
    ):
        super().__init__()
        
        self.task_weights = task_weights or {
            'ec': 1.0,
            'loc': 1.0,
            'func': 1.0,
        }
        
        self.use_pos_weight = use_pos_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        pos_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多任务损失
        
        Args:
            predictions: 模型输出
            targets: 真实标签
            pos_weights: 正样本权重 (用于处理类别不平衡)
        
        Returns:
            total_loss, task_losses dict
        """
        device = predictions['ec'].device
        task_losses = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # EC Loss (多标签)
        ec_loss = self.bce(predictions['ec'], targets['ec'])
        if self.use_pos_weight and pos_weights is not None:
            ec_loss = ec_loss * pos_weights['ec'].to(device)
        ec_loss = ec_loss.mean()
        task_losses['ec'] = ec_loss
        total_loss += self.task_weights['ec'] * ec_loss
        
        # Location Loss (多分类)
        loc_loss = self.ce(predictions['loc'], targets['loc'])
        loc_loss = loc_loss.mean()
        task_losses['loc'] = loc_loss
        total_loss += self.task_weights['loc'] * loc_loss
        
        # Function Loss (多标签)
        func_loss = self.bce(predictions['func'], targets['func'])
        if self.use_pos_weight and pos_weights is not None:
            func_loss = func_loss * pos_weights['func'].to(device)
        func_loss = func_loss.mean()
        task_losses['func'] = func_loss
        total_loss += self.task_weights['func'] * func_loss
        
        return total_loss, task_losses


class HierarchicalECClassifier(nn.Module):
    """层级 EC 分类器
    
    预测 EC 号的层级结构:
    - Level 1: 主类 (1-6)
    - Level 2: 亚类 (1.x)
    - Level 3: 子类 (1.x.x)
    - Level 4: 完整 EC (1.x.x.x)
    """
    
    def __init__(
        self,
        input_dim: int,
        n_classes_per_level: List[int] = [6, 50, 200, 1000],
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # 共享编码器
        self.encoder = SharedFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=[512, 256],
            dropout=dropout,
        )
        
        # 每层的分类头
        self.heads = nn.ModuleList()
        prev_dim = self.encoder.output_dim
        
        for n_classes in n_classes_per_level:
            self.heads.append(nn.Sequential(
                nn.Linear(prev_dim, n_classes),
            ))
            prev_dim = n_classes  # 层级间信息传递
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回每层的预测"""
        features = self.encoder(x)
        
        outputs = []
        current_input = features
        
        for head in self.heads:
            output = head(current_input)
            outputs.append(output)
            # 使用预测作为下一层的额外输入
            if self.training and len(outputs) < len(self.heads):
                current_input = torch.cat([features, F.softmax(output, dim=-1)], dim=-1)
            else:
                current_input = torch.cat([features, F.softmax(output.detach(), dim=-1)], dim=-1)
        
        return outputs


def create_model(
    input_dim: int,
    ec_num_classes: int,
    loc_num_classes: int,
    func_num_classes: int,
    model_type: str = 'multi_task',
    **kwargs
) -> nn.Module:
    """模型创建工厂"""
    
    if model_type == 'multi_task':
        return MultiTaskProteinClassifier(
            input_dim=input_dim,
            ec_num_classes=ec_num_classes,
            loc_num_classes=loc_num_classes,
            func_num_classes=func_num_classes,
            **kwargs
        )
    elif model_type == 'hierarchical_ec':
        return HierarchicalECClassifier(
            input_dim=input_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试模型
    batch_size = 16
    input_dim = 320  # ESM2 8M
    ec_classes = 500
    loc_classes = 30
    func_classes = 50
    
    # 创建模型
    model = MultiTaskProteinClassifier(
        input_dim=input_dim,
        ec_num_classes=ec_classes,
        loc_num_classes=loc_classes,
        func_num_classes=func_classes,
    )
    
    # 测试输入
    x = torch.randn(batch_size, input_dim)
    outputs = model(x)
    
    print("模型输出:")
    print(f"  EC logits: {outputs['ec'].shape}")
    print(f"  Location logits: {outputs['loc'].shape}")
    print(f"  Function logits: {outputs['func'].shape}")
    
    # 测试损失
    targets = {
        'ec': torch.randint(0, 2, (batch_size, ec_classes)).float(),
        'loc': torch.randint(0, loc_classes, (batch_size,)),
        'func': torch.randint(0, 2, (batch_size, func_classes)).float(),
    }
    
    criterion = MultiTaskLoss()
    loss, task_losses = criterion(outputs, targets)
    
    print(f"\n损失:")
    print(f"  Total: {loss.item():.4f}")
    for task, loss_val in task_losses.items():
        print(f"  {task}: {loss_val.item():.4f}")
