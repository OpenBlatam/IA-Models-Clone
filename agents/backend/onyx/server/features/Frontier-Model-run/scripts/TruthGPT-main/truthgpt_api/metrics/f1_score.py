"""
F1 Score Metric for TruthGPT API
===============================

TensorFlow-like F1 score metric implementation.
"""

import torch
from typing import Optional, Union
from .precision import Precision
from .recall import Recall


class F1Score:
    """
    F1 score metric.
    
    Similar to tf.keras.metrics.F1Score, this metric
    computes the F1 score of predictions.
    """
    
    def __init__(self, 
                 thresholds: Union[float, list] = 0.5,
                 top_k: Optional[int] = None,
                 class_id: Optional[int] = None,
                 average: str = 'macro',
                 name: Optional[str] = None):
        """
        Initialize F1 score metric.
        
        Args:
            thresholds: Threshold values for binary classification
            top_k: Number of top predictions to consider
            class_id: Specific class ID to compute F1 score for
            average: Averaging method ('macro', 'micro', 'weighted')
            name: Optional name for the metric
        """
        self.thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        self.average = average
        self.name = name or "f1_score"
        
        # Create precision and recall metrics
        self.precision = Precision(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id
        )
        self.recall = Recall(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id
        )
        
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.precision.reset()
        self.recall.reset()
        self.total_samples = 0
    
    def update_state(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Update metric state.
        
        Args:
            y_true: True labels
            y_pred: Predictions
        """
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)
        self.total_samples += y_true.size(0)
    
    def result(self) -> float:
        """
        Get metric result.
        
        Returns:
            F1 score value
        """
        precision = self.precision.result()
        recall = self.recall.result()
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Compute F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            
        Returns:
            F1 score value
        """
        self.update_state(y_true, y_pred)
        return self.result()
    
    def __repr__(self):
        return f"F1Score(thresholds={self.thresholds}, top_k={self.top_k}, class_id={self.class_id}, average={self.average})"









