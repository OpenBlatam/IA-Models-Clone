"""
Precision Metric for TruthGPT API
=================================

TensorFlow-like precision metric implementation.
"""

import torch
from typing import Optional, Union


class Precision:
    """
    Precision metric.
    
    Similar to tf.keras.metrics.Precision, this metric
    computes the precision of predictions.
    """
    
    def __init__(self, 
                 thresholds: Union[float, list] = 0.5,
                 top_k: Optional[int] = None,
                 class_id: Optional[int] = None,
                 name: Optional[str] = None):
        """
        Initialize precision metric.
        
        Args:
            thresholds: Threshold values for binary classification
            top_k: Number of top predictions to consider
            class_id: Specific class ID to compute precision for
            name: Optional name for the metric
        """
        self.thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        self.name = name or "precision"
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.true_positives = 0
        self.false_positives = 0
        self.total_samples = 0
    
    def update_state(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Update metric state.
        
        Args:
            y_true: True labels
            y_pred: Predictions
        """
        if self.class_id is not None:
            # Compute precision for specific class
            y_true_class = (y_true == self.class_id).float()
            y_pred_class = (y_pred == self.class_id).float()
        else:
            # Compute precision for all classes
            y_true_class = y_true.float()
            y_pred_class = y_pred.float()
        
        # Apply threshold
        if isinstance(self.thresholds, (int, float)):
            y_pred_binary = (y_pred_class > self.thresholds).float()
        else:
            # Multiple thresholds
            y_pred_binary = torch.zeros_like(y_pred_class)
            for threshold in self.thresholds:
                y_pred_binary += (y_pred_class > threshold).float()
            y_pred_binary = torch.clamp(y_pred_binary, 0, 1)
        
        # Compute true positives and false positives
        true_positives = (y_true_class * y_pred_binary).sum().item()
        false_positives = ((1 - y_true_class) * y_pred_binary).sum().item()
        
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.total_samples += y_true.size(0)
    
    def result(self) -> float:
        """
        Get metric result.
        
        Returns:
            Precision value
        """
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Compute precision.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            
        Returns:
            Precision value
        """
        self.update_state(y_true, y_pred)
        return self.result()
    
    def __repr__(self):
        return f"Precision(thresholds={self.thresholds}, top_k={self.top_k}, class_id={self.class_id})"


