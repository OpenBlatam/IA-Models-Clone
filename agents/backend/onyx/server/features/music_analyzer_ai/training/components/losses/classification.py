"""
Classification Loss Functions

Specialized loss functions for classification tasks.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Classification loss with label smoothing support.
    
    Uses CrossEntropyLoss with optional label smoothing
    for better generalization.
    """
    
    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        """
        Initialize classification loss.
        
        Args:
            num_classes: Number of classes
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            weight: Optional class weights tensor
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=weight,
            reduction=reduction
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            predictions: Model predictions [batch, num_classes]
            targets: Target labels [batch]
            
        Returns:
            Loss tensor
        """
        return self.criterion(predictions, targets)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focuses learning on hard examples by down-weighting
    easy examples.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Model predictions [batch, num_classes]
            targets: Target labels [batch]
            
        Returns:
            Loss tensor
        """
        ce_loss = F.cross_entropy(
            predictions,
            targets,
            reduction="none"
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for regularization.
    
    Smooths target distribution to prevent overconfidence.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = "mean"
    ):
        """
        Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            predictions: Model predictions [batch, num_classes]
            targets: Target labels [batch]
            
        Returns:
            Loss tensor
        """
        log_probs = F.log_softmax(predictions, dim=-1)
        
        # Create smoothed targets
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute KL divergence
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


__all__ = [
    "ClassificationLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
]



