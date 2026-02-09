"""
Real-time Metrics for Flux2 Clothing Changer
============================================

Real-time metrics collection and streaming.
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
    metric_name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class RealTimeMetrics:
    """Real-time metrics system."""
    
    def __init__(
        self,
        window_size: int = 1000,
        update_interval: float = 1.0,
    ):
        """
        Initialize real-time metrics.
        
        Args:
            window_size: Metrics window size
            update_interval: Update interval in seconds
        """
        self.window_size = window_size
        self.update_interval = update_interval
        
        self.metrics: Dict[str, deque] = {}
        self.subscribers: List[Callable] = []
        self.last_update = time.time()
    
    def subscribe(self, callback: Callable[[List[MetricPoint]], None]) -> None:
        """
        Subscribe to metric updates.
        
        Args:
            callback: Callback function
        """
        self.subscribers.append(callback)
        logger.info("Subscribed to real-time metrics")
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record metric.
        
        Args:
            metric_name: Metric name
            value: Metric value
            tags: Optional tags
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.window_size)
        
        point = MetricPoint(
            metric_name=metric_name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
        )
        
        self.metrics[metric_name].append(point)
        
        # Notify subscribers if interval passed
        if time.time() - self.last_update >= self.update_interval:
            self._notify_subscribers()
            self.last_update = time.time()
    
    def _notify_subscribers(self) -> None:
        """Notify subscribers of recent metrics."""
        recent_points = []
        
        for metric_name, points in self.metrics.items():
            if points:
                recent_points.append(points[-1])
        
        for callback in self.subscribers:
            try:
                callback(recent_points)
            except Exception as e:
                logger.error(f"Error in metrics subscriber: {e}")
    
    def get_metric_values(
        self,
        metric_name: str,
        time_range: Optional[float] = None,
    ) -> List[float]:
        """
        Get metric values.
        
        Args:
            metric_name: Metric name
            time_range: Optional time range in seconds
            
        Returns:
            List of metric values
        """
        if metric_name not in self.metrics:
            return []
        
        points = list(self.metrics[metric_name])
        
        if time_range:
            cutoff_time = time.time() - time_range
            points = [p for p in points if p.timestamp >= cutoff_time]
        
        return [p.value for p in points]
    
    def get_current_value(self, metric_name: str) -> Optional[float]:
        """Get current metric value."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1].value
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get metrics statistics."""
        return {
            "total_metrics": len(self.metrics),
            "metric_names": list(self.metrics.keys()),
            "subscribers": len(self.subscribers),
        }


