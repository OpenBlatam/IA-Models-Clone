"""
Performance Tracker for Flux2 Clothing Changer
==============================================

Advanced performance tracking and analysis.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric snapshot."""
    metric_name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class PerformanceTracker:
    """Advanced performance tracking system."""
    
    def __init__(
        self,
        history_size: int = 10000,
    ):
        """
        Initialize performance tracker.
        
        Args:
            history_size: Maximum number of metrics to keep
        """
        self.history_size = history_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.aggregations: Dict[str, Dict[str, float]] = {}
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record performance metric.
        
        Args:
            metric_name: Metric name
            value: Metric value
            tags: Optional tags
        """
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
        )
        
        self.metrics[metric_name].append(metric)
        
        # Update aggregations
        self._update_aggregations(metric_name, value)
    
    def get_metrics(
        self,
        metric_name: str,
        time_range: Optional[float] = None,
    ) -> List[PerformanceMetric]:
        """
        Get metrics for a metric name.
        
        Args:
            metric_name: Metric name
            time_range: Optional time range in seconds
            
        Returns:
            List of metrics
        """
        if metric_name not in self.metrics:
            return []
        
        metrics = list(self.metrics[metric_name])
        
        if time_range:
            cutoff_time = time.time() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return metrics
    
    def get_statistics(
        self,
        metric_name: str,
        time_range: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Metric name
            time_range: Optional time range in seconds
            
        Returns:
            Statistics dictionary
        """
        metrics = self.get_metrics(metric_name, time_range)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99),
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            metric_name: self.get_statistics(metric_name)
            for metric_name in self.metrics.keys()
        }
    
    def _update_aggregations(self, metric_name: str, value: float) -> None:
        """Update metric aggregations."""
        if metric_name not in self.aggregations:
            self.aggregations[metric_name] = {
                "count": 0,
                "sum": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }
        
        agg = self.aggregations[metric_name]
        agg["count"] += 1
        agg["sum"] += value
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_metrics": len(self.metrics),
            "metric_names": list(self.metrics.keys()),
            "total_data_points": sum(len(metrics) for metrics in self.metrics.values()),
        }


