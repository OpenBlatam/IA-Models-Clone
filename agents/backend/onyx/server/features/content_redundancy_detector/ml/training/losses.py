"""
Loss Functions Module
Modular loss function implementations and factories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss"""
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
    Label Smoothing Cross Entropy Loss
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize label smoothing loss
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss"""
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class LossFactory:
    """
    Factory for creating loss functions
    """
    
    @staticmethod
    def create_loss(
        loss_type: str,
        num_classes: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Create loss function
        
        Args:
            loss_type: Type of loss ('cross_entropy', 'focal', 'label_smoothing')
            num_classes: Number of classes (required for some losses)
            **kwargs: Additional loss parameters
            
        Returns:
            Loss function
        """
        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'focal':
            return FocalLoss(**kwargs)
        elif loss_type == 'label_smoothing':
            if num_classes is None:
                raise ValueError("num_classes required for label_smoothing loss")
            return LabelSmoothingLoss(num_classes=num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")



