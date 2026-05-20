"""
Modular Loss Functions
Various loss functions for different tasks
"""

from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class ClassificationLoss(nn.Module):
    """Classification loss with label smoothing"""
    
    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.weight = weight
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=weight
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification loss"""
        return self.criterion(predictions, targets)


class RegressionLoss(nn.Module):
    """Regression loss with optional Huber loss"""
    
    def __init__(
        self,
        loss_type: str = "mse",  # "mse", "mae", "huber"
        reduction: str = "mean"
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == "mae":
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == "huber":
            self.criterion = nn.HuberLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute regression loss"""
        return self.criterion(predictions, targets)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification"""
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss"""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss"""
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute label smoothing loss"""
        log_probs = F.log_softmax(predictions, dim=-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining multiple loss functions"""
    
    def __init__(
        self,
        task_losses: Dict[str, nn.Module],
        task_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.task_losses = nn.ModuleDict(task_losses)
        self.task_weights = task_weights or {task: 1.0 for task in task_losses.keys()}
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        losses = {}
        total_loss = 0.0
        
        for task_name, loss_fn in self.task_losses.items():
            if task_name in predictions and task_name in targets:
                task_loss = loss_fn(predictions[task_name], targets[task_name])
                weight = self.task_weights.get(task_name, 1.0)
                weighted_loss = weight * task_loss
                losses[task_name] = task_loss
                total_loss += weighted_loss
        
        losses["total"] = total_loss
        return losses



