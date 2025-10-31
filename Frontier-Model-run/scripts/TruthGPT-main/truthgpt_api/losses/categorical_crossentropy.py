"""
Categorical Crossentropy Loss for TruthGPT API
=============================================

TensorFlow-like categorical crossentropy loss implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Union


class CategoricalCrossentropy:
    """
    Categorical crossentropy loss.
    
    Similar to tf.keras.losses.CategoricalCrossentropy, this loss
    function computes the crossentropy loss between true labels and predictions.
    """
    
    def __init__(self, 
                 from_logits: bool = False,
                 label_smoothing: float = 0.0,
                 reduction: str = 'auto',
                 name: Optional[str] = None):
        """
        Initialize categorical crossentropy loss.
        
        Args:
            from_logits: Whether predictions are logits
            label_smoothing: Label smoothing factor
            reduction: Reduction method
            name: Optional name for the loss
        """
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.name = name or "categorical_crossentropy"
        
        # Create PyTorch loss function
        if from_logits:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                reduction='mean' if reduction == 'auto' else reduction
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                reduction='mean' if reduction == 'auto' else reduction
            )
    
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            
        Returns:
            Loss value
        """
        if not self.from_logits:
            # Apply softmax to predictions
            y_pred = torch.softmax(y_pred, dim=-1)
        
        # Convert to class indices if needed
        if y_true.dim() > 1 and y_true.size(-1) > 1:
            y_true = torch.argmax(y_true, dim=-1)
        
        return self.loss_fn(y_pred, y_true)
    
    def get_config(self) -> dict:
        """Get loss configuration."""
        return {
            'name': self.name,
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing,
            'reduction': self.reduction
        }
    
    def __repr__(self):
        return f"CategoricalCrossentropy(from_logits={self.from_logits}, label_smoothing={self.label_smoothing})"


class SparseCategoricalCrossentropy:
    """
    Sparse categorical crossentropy loss.
    
    Similar to tf.keras.losses.SparseCategoricalCrossentropy, this loss
    function computes the crossentropy loss for sparse labels.
    """
    
    def __init__(self, 
                 from_logits: bool = False,
                 reduction: str = 'auto',
                 name: Optional[str] = None):
        """
        Initialize sparse categorical crossentropy loss.
        
        Args:
            from_logits: Whether predictions are logits
            reduction: Reduction method
            name: Optional name for the loss
        """
        self.from_logits = from_logits
        self.reduction = reduction
        self.name = name or "sparse_categorical_crossentropy"
        
        # Create PyTorch loss function
        self.loss_fn = nn.CrossEntropyLoss(
            reduction='mean' if reduction == 'auto' else reduction
        )
    
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            y_true: True labels (sparse)
            y_pred: Predictions
            
        Returns:
            Loss value
        """
        if not self.from_logits:
            # Apply softmax to predictions
            y_pred = torch.softmax(y_pred, dim=-1)
        
        return self.loss_fn(y_pred, y_true)
    
    def get_config(self) -> dict:
        """Get loss configuration."""
        return {
            'name': self.name,
            'from_logits': self.from_logits,
            'reduction': self.reduction
        }
    
    def __repr__(self):
        return f"SparseCategoricalCrossentropy(from_logits={self.from_logits})"









