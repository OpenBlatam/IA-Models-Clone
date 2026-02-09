"""
Real-time Metrics V2 for Flux2 Clothing Changer
=================================================

Advanced real-time metrics with streaming and aggregation.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class MetricWindow:
    """Metric window aggregation."""
    metric_name: str
    window_size: int
    points: deque
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    
    def __post_init__(self):
        if not hasattr(self, 'points'):
            self.points = deque(maxlen=self.window_size)


class RealTimeMetricsV2:
    """Advanced real-time metrics system."""
    
    def __init__(
        self,
        window_size: int = 1000,
        aggregation_interval: float = 1.0,
    ):
        """
        Initialize real-time metrics.
        
        Args:
            window_size: Size of metric windows
            aggregation_interval: Aggregation interval in seconds
        """
        self.window_size = window_size
        self.aggregation_interval = aggregation_interval
        
        self.metrics: Dict[str, MetricWindow] = {}
        self.subscribers: List[Callable] = []
        self.last_aggregation = time.time()
    
    def record(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record metric value.
        
        Args:
            metric_name: Metric name
            value: Metric value
            tags: Optional tags
        """
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {},
        )
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = MetricWindow(
                metric_name=metric_name,
                window_size=self.window_size,
                points=deque(maxlen=self.window_size),
            )
        
        window = self.metrics[metric_name]
        window.points.append(point)
        
        # Update aggregations
        window.sum += value
        window.min = min(window.min, value)
        window.max = max(window.max, value)
        
        # Remove old points from sum
        if len(window.points) == window.window_size:
            old_point = window.points[0]
            window.sum -= old_point.value
        
        # Notify subscribers
        self._notify_subscribers(metric_name, point)
    
    def _notify_subscribers(self, metric_name: str, point: MetricPoint) -> None:
        """
        Notify subscribers of new metric.
        
        Args:
            metric_name: Metric name
            point: Metric point
        """
        for subscriber in self.subscribers:
            try:
                subscriber(metric_name, point)
            except Exception as e:
                logger.error(f"Subscriber notification failed: {e}")
    
    def subscribe(self, callback: Callable) -> None:
        """
        Subscribe to metric updates.
        
        Args:
            callback: Callback function
        """
        self.subscribers.append(callback)
        logger.info(f"Subscribed to metrics: {callback.__name__}")
    
    def get_metric(
        self,
        metric_name: str,
        aggregation: str = "mean",
    ) -> Optional[float]:
        """
        Get aggregated metric value.
        
        Args:
            metric_name: Metric name
            aggregation: Aggregation type (mean, sum, min, max, count)
            
        Returns:
            Aggregated value or None
        """
        if metric_name not in self.metrics:
            return None
        
        window = self.metrics[metric_name]
        
        if not window.points:
            return None
        
        if aggregation == "mean":
            return window.sum / len(window.points)
        elif aggregation == "sum":
            return window.sum
        elif aggregation == "min":
            return window.min
        elif aggregation == "max":
            return window.max
        elif aggregation == "count":
            return len(window.points)
        else:
            return window.sum / len(window.points)
    
    def get_metric_window(
        self,
        metric_name: str,
        time_range: Optional[float] = None,
    ) -> List[MetricPoint]:
        """
        Get metric points in time range.
        
        Args:
            metric_name: Metric name
            time_range: Time range in seconds
            
        Returns:
            List of metric points
        """
        if metric_name not in self.metrics:
            return []
        
        window = self.metrics[metric_name]
        points = list(window.points)
        
        if time_range:
            cutoff_time = time.time() - time_range
            points = [p for p in points if p.timestamp >= cutoff_time]
        
        return points
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all metrics with aggregations.
        
        Returns:
            Dictionary of metrics
        """
        result = {}
        
        for metric_name, window in self.metrics.items():
            if not window.points:
                continue
            
            result[metric_name] = {
                "count": len(window.points),
                "mean": window.sum / len(window.points),
                "sum": window.sum,
                "min": window.min,
                "max": window.max,
                "latest": window.points[-1].value if window.points else None,
            }
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get metrics system statistics."""
        return {
            "total_metrics": len(self.metrics),
            "subscribers": len(self.subscribers),
            "window_size": self.window_size,
            "aggregation_interval": self.aggregation_interval,
        }


