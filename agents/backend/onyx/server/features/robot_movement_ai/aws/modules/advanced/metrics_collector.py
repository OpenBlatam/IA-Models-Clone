"""
Advanced Metrics Collector
==========================

Advanced metrics collection and analysis.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Metric data point."""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class AdvancedMetricsCollector:
    """Advanced metrics collector with analysis."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._metrics: Dict[str, deque] = {}
        self._aggregations: Dict[str, Dict[str, float]] = {}
    
    def record(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record metric value."""
        if metric_name not in self._metrics:
            self._metrics[metric_name] = deque(maxlen=self.window_size)
        
        point = MetricPoint(value=value, tags=tags or {})
        self._metrics[metric_name].append(point)
        
        # Update aggregations
        self._update_aggregations(metric_name)
    
    def _update_aggregations(self, metric_name: str):
        """Update metric aggregations."""
        if metric_name not in self._metrics:
            return
        
        values = [point.value for point in self._metrics[metric_name]]
        
        if not values:
            return
        
        self._aggregations[metric_name] = {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99)
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get metric statistics."""
        if metric_name not in self._metrics:
            return None
        
        aggregations = self._aggregations.get(metric_name, {})
        
        return {
            "name": metric_name,
            "aggregations": aggregations,
            "recent_values": [
                {"value": point.value, "timestamp": point.timestamp.isoformat()}
                for point in list(self._metrics[metric_name])[-10:]
            ]
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "total_metrics": len(self._metrics),
            "metrics": {
                name: self._aggregations.get(name, {})
                for name in self._metrics.keys()
            }
        }
    
    def get_trend(self, metric_name: str, window_minutes: int = 5) -> Optional[Dict[str, Any]]:
        """Get metric trend."""
        if metric_name not in self._metrics:
            return None
        
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = [
            point for point in self._metrics[metric_name]
            if point.timestamp > cutoff
        ]
        
        if len(recent) < 2:
            return None
        
        values = [point.value for point in recent]
        
        # Calculate trend (simple linear regression)
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        
        return {
            "trend": trend,
            "slope": slope,
            "change_percent": (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0,
            "samples": n
        }















