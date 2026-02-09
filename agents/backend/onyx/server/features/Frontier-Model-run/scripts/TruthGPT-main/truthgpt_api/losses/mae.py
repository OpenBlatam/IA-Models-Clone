"""
Mean Absolute Error Loss for TruthGPT API
=========================================

TensorFlow-like MAE loss implementation.
"""

import torch
import torch.nn as nn
from typing import Optional


class MeanAbsoluteError:
    """
    Mean absolute error loss.
    
    Similar to tf.keras.losses.MeanAbsoluteError, this loss
    function computes the mean absolute error between true values and predictions.
    """
    
    def __init__(self, 
                 reduction: str = 'auto',
                 name: Optional[str] = None):
        """
        Initialize mean absolute error loss.
        
        Args:
            reduction: Reduction method
            name: Optional name for the loss
        """
        self.reduction = reduction
        self.name = name or "mean_absolute_error"
        
        # Create PyTorch loss function
        self.loss_fn = nn.L1Loss(
            reduction='mean' if reduction == 'auto' else reduction
        )
    
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            y_true: True values
            y_pred: Predictions
            
        Returns:
            Loss value
        """
        return self.loss_fn(y_pred, y_true)
    
    def get_config(self) -> dict:
        """Get loss configuration."""
        return {
            'name': self.name,
            'reduction': self.reduction
        }
    
    def __repr__(self):
        return f"MeanAbsoluteError(reduction={self.reduction})"









