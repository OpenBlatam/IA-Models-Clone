"""
Custom Loss Functions
=====================

Advanced loss functions for various tasks.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        Forward pass.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss
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
    Label Smoothing Loss.
    
    Regularization technique for classification.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)
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
            smooth: Smoothing factor
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Model predictions (probabilities)
            targets: Ground truth (binary)
            
        Returns:
            Dice loss
        """
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss function (weighted sum of multiple losses).
    """
    
    def __init__(self, losses: list, weights: Optional[list] = None):
        """
        Initialize Combined Loss.
        
        Args:
            losses: List of loss functions
            weights: Weights for each loss (defaults to equal weights)
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        
        if len(weights) != len(losses):
            raise ValueError("Number of weights must match number of losses")
        
        self.weights = weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Model predictions
            targets: Ground truth
            
        Returns:
            Combined loss
        """
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        
        return total_loss


def create_loss(
    loss_type: str,
    num_classes: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating loss functions.
    
    Args:
        loss_type: Type of loss ('ce', 'focal', 'smooth', 'dice', 'mse', 'mae')
        num_classes: Number of classes (for classification losses)
        **kwargs: Additional loss-specific parameters
        
    Returns:
        Loss function
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'ce' or loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 1.0),
            gamma=kwargs.get('gamma', 2.0),
            **{k: v for k, v in kwargs.items() if k not in ['alpha', 'gamma']}
        )
    
    elif loss_type == 'smooth' or loss_type == 'label_smoothing':
        if num_classes is None:
            raise ValueError("num_classes required for label smoothing loss")
        return LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=kwargs.get('smoothing', 0.1)
        )
    
    elif loss_type == 'dice':
        return DiceLoss(smooth=kwargs.get('smooth', 1.0))
    
    elif loss_type == 'mse' or loss_type == 'mean_squared_error':
        return nn.MSELoss(**kwargs)
    
    elif loss_type == 'mae' or loss_type == 'mean_absolute_error':
        return nn.L1Loss(**kwargs)
    
    elif loss_type == 'bce' or loss_type == 'binary_cross_entropy':
        return nn.BCEWithLogitsLoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")



