"""
Loss Functions for Music Generation

Implements:
- Various loss functions for audio generation
- Combined losses
- Loss weighting strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class MSELoss(nn.Module):
    """Mean Squared Error loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction=reduction)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        return self.criterion(predictions, targets)


class MAELoss(nn.Module):
    """Mean Absolute Error loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.criterion = nn.L1Loss(reduction=reduction)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MAE loss."""
        return self.criterion(predictions, targets)


class SpectralLoss(nn.Module):
    """Spectral loss for audio generation."""
    
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss."""
        # Compute STFT
        pred_stft = torch.stft(
            predictions.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )
        target_stft = torch.stft(
            targets.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )
        
        # Compute magnitude
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        # L1 loss on magnitude
        loss = F.l1_loss(pred_mag, target_mag)
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss with multiple components."""
    
    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize combined loss.
        
        Args:
            losses: Dictionary of loss functions
            weights: Optional weights for each loss
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses.keys()}
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Returns:
            Dictionary with individual losses and total loss
        """
        loss_dict = {}
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(predictions, targets)
            weighted_loss = loss_value * self.weights.get(name, 1.0)
            loss_dict[name] = loss_value
            total_loss += weighted_loss
        
        loss_dict['total'] = total_loss
        
        return loss_dict


def create_loss_function(
    loss_type: str = "mse",
    **kwargs
) -> nn.Module:
    """
    Create loss function.
    
    Args:
        loss_type: Type of loss ('mse', 'mae', 'spectral', etc.)
        **kwargs: Loss-specific parameters
        
    Returns:
        Loss function
    """
    loss_map = {
        'mse': MSELoss,
        'mae': MAELoss,
        'l1': MAELoss,
        'spectral': SpectralLoss
    }
    
    if loss_type.lower() not in loss_map:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: {list(loss_map.keys())}"
        )
    
    loss_class = loss_map[loss_type.lower()]
    return loss_class(**kwargs)



