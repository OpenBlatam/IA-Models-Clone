"""
Metrics Aggregator
==================

Advanced metrics aggregation with time windows and percentiles.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)

@dataclass
class MetricDataPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsAggregator:
    """Advanced metrics aggregator."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.aggregated: Dict[str, Dict[str, Any]] = {}
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        
        data_point = MetricDataPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        
        self.metrics[name].append(data_point)
    
    def get_aggregated(self, name: str, window: Optional[int] = None) -> Dict[str, Any]:
        """Get aggregated statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        data_points = list(self.metrics[name])
        if window:
            data_points = data_points[-window:]
        
        if not data_points:
            return {}
        
        values = [dp.value for dp in data_points]
        
        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p50": statistics.median(values),
            "p75": self._percentile(values, 0.75),
            "p90": self._percentile(values, 0.90),
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99),
            "p999": self._percentile(values, 0.999),
            "latest": values[-1] if values else None,
            "first": values[0] if values else None
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all aggregated metrics."""
        return {
            name: self.get_aggregated(name)
            for name in self.metrics.keys()
        }
    
    def get_time_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        if name not in self.metrics:
            return []
        
        data_points = list(self.metrics[name])
        
        if start_time:
            data_points = [dp for dp in data_points if dp.timestamp >= start_time]
        if end_time:
            data_points = [dp for dp in data_points if dp.timestamp <= end_time]
        
        return [
            {
                "timestamp": dp.timestamp.isoformat(),
                "value": dp.value,
                "tags": dp.tags
            }
            for dp in data_points
        ]
    
    def reset(self, name: Optional[str] = None):
        """Reset metrics."""
        if name:
            if name in self.metrics:
                self.metrics[name].clear()
        else:
            self.metrics.clear()
            self.aggregated.clear()

# Global instance
metrics_aggregator = MetricsAggregator()
































