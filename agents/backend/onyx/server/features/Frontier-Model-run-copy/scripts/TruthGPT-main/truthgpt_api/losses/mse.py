"""
Mean Squared Error Loss for TruthGPT API
=======================================

TensorFlow-like MSE loss implementation.
"""

import torch
import torch.nn as nn
from typing import Optional


class MeanSquaredError:
    """
    Mean squared error loss.
    
    Similar to tf.keras.losses.MeanSquaredError, this loss
    function computes the mean squared error between true values and predictions.
    """
    
    def __init__(self, 
                 reduction: str = 'auto',
                 name: Optional[str] = None):
        """
        Initialize mean squared error loss.
        
        Args:
            reduction: Reduction method
            name: Optional name for the loss
        """
        self.reduction = reduction
        self.name = name or "mean_squared_error"
        
        # Create PyTorch loss function
        self.loss_fn = nn.MSELoss(
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
        return f"MeanSquaredError(reduction={self.reduction})"


