"""
Custom Loss Functions
====================

Custom loss functions for various deep learning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for regularization.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0.0-1.0)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
        
        Returns:
            Label smoothing loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Model predictions (probabilities)
            targets: Ground truth (one-hot encoded)
        
        Returns:
            Dice loss value
        """
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss function (e.g., CrossEntropy + Dice).
    """
    
    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize combined loss.
        
        Args:
            losses: Dictionary of loss functions
            weights: Weights for each loss (defaults to equal weights)
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 / len(losses) for name in losses.keys()}
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            inputs: Model predictions
            targets: Ground truth
        
        Returns:
            Dictionary with individual and total losses
        """
        results = {}
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(inputs, targets)
            weighted_loss = self.weights[name] * loss_value
            results[name] = loss_value
            results[f"{name}_weighted"] = weighted_loss
            total_loss += weighted_loss
        
        results['total'] = total_loss
        return results


def get_loss_function(
    loss_name: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    
    Examples:
        >>> loss_fn = get_loss_function("focal", alpha=1.0, gamma=2.0)
        >>> loss_fn = get_loss_function("label_smoothing", num_classes=10, smoothing=0.1)
        >>> loss_fn = get_loss_function("cross_entropy")
    """
    loss_name = loss_name.lower()
    
    if loss_name == "cross_entropy" or loss_name == "ce":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == "mse" or loss_name == "mean_squared_error":
        return nn.MSELoss(**kwargs)
    elif loss_name == "mae" or loss_name == "mean_absolute_error":
        return nn.L1Loss(**kwargs)
    elif loss_name == "bce" or loss_name == "binary_cross_entropy":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == "focal":
        return FocalLoss(**kwargs)
    elif loss_name == "label_smoothing":
        return LabelSmoothingLoss(**kwargs)
    elif loss_name == "dice":
        return DiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")



