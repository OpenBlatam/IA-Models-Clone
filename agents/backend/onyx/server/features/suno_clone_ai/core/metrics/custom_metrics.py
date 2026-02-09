"""
Custom Metrics

Utilities for creating and registering custom metrics.
"""

import logging
from typing import Callable, Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Registry for custom metrics."""
    
    def __init__(self):
        """Initialize metric registry."""
        self.metrics: Dict[str, Callable] = {}
    
    def register(
        self,
        name: str,
        metric_fn: Callable
    ) -> None:
        """
        Register a custom metric.
        
        Args:
            name: Metric name
            metric_fn: Metric function
        """
        self.metrics[name] = metric_fn
        logger.info(f"Registered metric: {name}")
    
    def get(
        self,
        name: str
    ) -> Optional[Callable]:
        """
        Get metric function.
        
        Args:
            name: Metric name
            
        Returns:
            Metric function or None
        """
        return self.metrics.get(name)
    
    def list_metrics(self) -> List[str]:
        """List all registered metrics."""
        return list(self.metrics.keys())


# Global registry
_global_registry = MetricRegistry()


def create_custom_metric(
    name: str,
    metric_fn: Callable
) -> None:
    """
    Create and register custom metric.
    
    Args:
        name: Metric name
        metric_fn: Metric function (predictions, targets) -> float
    """
    _global_registry.register(name, metric_fn)


def register_metric(
    name: str,
    metric_fn: Callable
) -> None:
    """Register metric in global registry."""
    _global_registry.register(name, metric_fn)


def get_metric(name: str) -> Optional[Callable]:
    """Get metric from global registry."""
    return _global_registry.get(name)


# Common custom metrics
def accuracy_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy."""
    if predictions.dim() > 1:
        preds = predictions.argmax(dim=-1)
    else:
        preds = (predictions > 0.5).long()
    
    correct = (preds == targets).float().mean()
    return correct.item()


def f1_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute F1 score."""
    if predictions.dim() > 1:
        preds = predictions.argmax(dim=-1)
    else:
        preds = (predictions > 0.5).long()
    
    tp = ((preds == 1) & (targets == 1)).float().sum()
    fp = ((preds == 1) & (targets == 0)).float().sum()
    fn = ((preds == 0) & (targets == 1)).float().sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1.item()


# Register common metrics
register_metric("accuracy", accuracy_metric)
register_metric("f1", f1_metric)



