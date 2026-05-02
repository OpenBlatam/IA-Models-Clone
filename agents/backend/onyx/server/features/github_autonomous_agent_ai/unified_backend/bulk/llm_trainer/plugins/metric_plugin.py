"""
Metric Plugin Interface
=======================

Plugin interface for custom metrics.

Author: BUL System
Date: 2024
"""

from typing import Dict, Any, Callable
from transformers import EvalPrediction
from .base_plugin import BasePlugin


class MetricPlugin(BasePlugin):
    """
    Base class for metric plugins.
    
    Allows adding custom metrics to evaluation.
    
    Example:
        >>> class MyMetricPlugin(MetricPlugin):
        ...     def __init__(self):
        ...         super().__init__("my_metric", "1.0.0")
        ...     
        ...     def compute(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        ...         # Calculate your metric
        ...         return {"my_metric": 0.95}
    """
    
    @property
    def compute_function(self) -> Callable[[EvalPrediction], Dict[str, float]]:
        """
        Get the compute function for this metric.
        
        Returns:
            Function that computes the metric
        """
        return self.compute
    
    def compute(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute the metric. Override this method.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary with metric name and value
        """
        if not self.enabled:
            return {}
        return self._compute_metric(eval_pred)
    
    def _compute_metric(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Internal method to compute metric. Override this.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary with metric values
        """
        return {}

