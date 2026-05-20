"""
Accuracy Metric for TruthGPT API
================================

TensorFlow-like accuracy metric implementation.
"""

import torch
from typing import Optional


class Accuracy:
    """
    Accuracy metric.
    
    Similar to tf.keras.metrics.Accuracy, this metric
    computes the accuracy of predictions.
    """
    
    def __init__(self, 
                 name: Optional[str] = None):
        """
        Initialize accuracy metric.
        
        Args:
            name: Optional name for the metric
        """
        self.name = name or "accuracy"
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.correct = 0
        self.total = 0
    
    def update_state(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Update metric state.
        
        Args:
            y_true: True labels
            y_pred: Predictions
        """
        # Convert predictions to class indices
        if y_pred.dim() > 1 and y_pred.size(-1) > 1:
            y_pred = torch.argmax(y_pred, dim=-1)
        
        # Count correct predictions
        correct = (y_pred == y_true).sum().item()
        total = y_true.size(0)
        
        self.correct += correct
        self.total += total
    
    def result(self) -> float:
        """
        Get metric result.
        
        Returns:
            Accuracy value
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Compute accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            
        Returns:
            Accuracy value
        """
        self.update_state(y_true, y_pred)
        return self.result()
    
    def __repr__(self):
        return f"Accuracy(name={self.name})"









