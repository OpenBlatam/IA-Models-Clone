"""
Regression Loss Functions

Specialized loss functions for regression tasks.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    """
    Regression loss with optional Huber loss.
    
    Supports both MSE and Huber loss for robustness
    to outliers.
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        delta: float = 1.0,
        reduction: str = "mean"
    ):
        """
        Initialize regression loss.
        
        Args:
            loss_type: Type of loss ("mse", "mae", "huber")
            delta: Delta parameter for Huber loss
            reduction: Reduction method
        """
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta
        self.reduction = reduction
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == "mae":
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == "huber":
            self.criterion = nn.HuberLoss(delta=delta, reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regression loss.
        
        Args:
            predictions: Model predictions [batch, ...]
            targets: Target values [batch, ...]
            
        Returns:
            Loss tensor
        """
        return self.criterion(predictions, targets)


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss variant).
    
    Less sensitive to outliers than MSE.
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = "mean"
    ):
        """
        Initialize smooth L1 loss.
        
        Args:
            beta: Threshold parameter
            reduction: Reduction method
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.criterion = nn.SmoothL1Loss(beta=beta, reduction=reduction)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute smooth L1 loss."""
        return self.criterion(predictions, targets)


__all__ = [
    "RegressionLoss",
    "SmoothL1Loss",
]



