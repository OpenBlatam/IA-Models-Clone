"""
Metrics Tracker

Advanced metrics tracking and aggregation.
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and aggregate metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.metric_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def track(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track a metric.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            step: Optional step number
            metadata: Optional metadata
        """
        self.metrics[metric_name].append(value)
        
        entry = {
            'value': value,
            'step': step,
            **(metadata or {})
        }
        self.metric_history[metric_name].append(entry)
    
    def get_metric(
        self,
        metric_name: str
    ) -> List[float]:
        """
        Get metric values.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            List of metric values
        """
        return self.metrics.get(metric_name, [])
    
    def get_statistics(
        self,
        metric_name: str
    ) -> Dict[str, float]:
        """
        Get metric statistics.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Statistics dictionary
        """
        values = self.metrics.get(metric_name, [])
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def aggregate(
        self,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics.
        
        Args:
            metric_names: List of metric names (None = all)
            
        Returns:
            Aggregated statistics
        """
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        
        aggregated = {}
        for name in metric_names:
            aggregated[name] = self.get_statistics(name)
        
        return aggregated
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.metric_history.clear()
        logger.info("Metrics tracker reset")


def track_metric(
    metric_name: str,
    value: float,
    **kwargs
) -> None:
    """Track a metric."""
    tracker = MetricsTracker()
    tracker.track(metric_name, value, **kwargs)


def aggregate_metrics(
    metrics: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from dictionary.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Aggregated statistics
    """
    tracker = MetricsTracker()
    for name, values in metrics.items():
        for value in values:
            tracker.track(name, value)
    
    return tracker.aggregate()


def compute_average_metrics(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Compute average metrics across multiple runs.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Average metrics
    """
    if not metrics_list:
        return {}
    
    # Get all metric names
    all_names = set()
    for metrics in metrics_list:
        all_names.update(metrics.keys())
    
    # Compute averages
    averages = {}
    for name in all_names:
        values = [m.get(name, 0.0) for m in metrics_list if name in m]
        if values:
            averages[name] = np.mean(values)
    
    return averages



