"""
Loss functions for training
Separated for modularity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConditionLoss(nn.Module):
    """
    Loss for condition classification (multi-label)
    Uses BCE with logits for multi-label classification
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super(ConditionLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            predictions: Model predictions (batch, num_conditions)
            targets: Ground truth (batch, num_conditions)
            
        Returns:
            Loss value
        """
        return F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class MetricLoss(nn.Module):
    """
    Loss for metric regression
    Uses MSE or L1 loss
    """
    
    def __init__(
        self,
        loss_type: str = "mse",  # "mse", "l1", "smooth_l1"
        reduction: str = 'mean'
    ):
        super(MetricLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            predictions: Model predictions (batch, num_metrics)
            targets: Ground truth (batch, num_metrics)
            
        Returns:
            Loss value
        """
        return self.loss_fn(predictions, targets)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining condition and metric losses
    Allows weighted combination of different tasks
    """
    
    def __init__(
        self,
        condition_weight: float = 1.0,
        metric_weight: float = 1.0,
        condition_loss_type: str = "bce",
        metric_loss_type: str = "mse",
        condition_pos_weight: Optional[torch.Tensor] = None
    ):
        super(MultiTaskLoss, self).__init__()
        
        self.condition_weight = condition_weight
        self.metric_weight = metric_weight
        
        # Initialize loss functions
        self.condition_loss = ConditionLoss(
            pos_weight=condition_pos_weight
        )
        self.metric_loss = MetricLoss(loss_type=metric_loss_type)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dictionary with 'conditions' and 'metrics'
            targets: Dictionary with 'conditions' and 'metrics'
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Condition loss
        if 'conditions' in predictions and 'conditions' in targets:
            condition_loss = self.condition_loss(
                predictions['conditions'],
                targets['conditions']
            )
            losses['condition_loss'] = condition_loss
        else:
            losses['condition_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # Metric loss
        if 'metrics' in predictions and 'metrics' in targets:
            metric_loss = self.metric_loss(
                predictions['metrics'],
                targets['metrics']
            )
            losses['metric_loss'] = metric_loss
        else:
            losses['metric_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # Total loss
        total_loss = (
            self.condition_weight * losses['condition_loss'] +
            self.metric_weight * losses['metric_loss']
        )
        losses['total_loss'] = total_loss
        
        return losses


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Useful for rare skin conditions
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            predictions: Model predictions (batch, num_classes)
            targets: Ground truth (batch, num_classes)
            
        Returns:
            Loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction='none'
        )
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    Can be adapted for multi-label classification
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dice loss
        
        Args:
            predictions: Model predictions (batch, num_classes)
            targets: Ground truth (batch, num_classes)
            
        Returns:
            Loss value
        """
        # Apply sigmoid to predictions
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice













