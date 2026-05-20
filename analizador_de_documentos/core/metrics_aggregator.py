"""
Metrics Aggregator for Document Analyzer
=========================================

Advanced metrics aggregation with percentiles and time series.
"""

import logging
import time
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsAggregator:
    """Advanced metrics aggregator"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        logger.info(f"MetricsAggregator initialized. Window size: {window_size}")
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric"""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        self.metrics[name].append(point)
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter"""
        self.counters[name] += value
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value"""
        self.gauges[name] = value
    
    def get_stats(self, name: str, window: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics for a metric"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return {}
        
        points = list(self.metrics[name])
        if window:
            points = points[-window:]
        
        values = [p.value for p in points]
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p50": sorted_values[int(len(sorted_values) * 0.5)],
            "p75": sorted_values[int(len(sorted_values) * 0.75)],
            "p90": sorted_values[int(len(sorted_values) * 0.90)],
            "p95": sorted_values[int(len(sorted_values) * 0.95)],
            "p99": sorted_values[int(len(sorted_values) * 0.99)],
            "p999": sorted_values[int(len(sorted_values) * 0.999)] if len(sorted_values) > 1000 else sorted_values[-1]
        }
    
    def get_time_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: timedelta = timedelta(seconds=60)
    ) -> List[Dict[str, Any]]:
        """Get time series data"""
        if name not in self.metrics:
            return []
        
        points = list(self.metrics[name])
        
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        # Aggregate by interval
        if not points:
            return []
        
        series = []
        current_bucket_start = points[0].timestamp
        current_bucket = []
        
        for point in points:
            if point.timestamp >= current_bucket_start + interval:
                if current_bucket:
                    values = [p.value for p in current_bucket]
                    series.append({
                        "timestamp": current_bucket_start.isoformat(),
                        "count": len(values),
                        "avg": statistics.mean(values),
                        "min": min(values),
                        "max": max(values)
                    })
                current_bucket_start = point.timestamp
                current_bucket = []
            
            current_bucket.append(point)
        
        # Add last bucket
        if current_bucket:
            values = [p.value for p in current_bucket]
            series.append({
                "timestamp": current_bucket_start.isoformat(),
                "count": len(values),
                "avg": statistics.mean(values),
                "min": min(values),
                "max": max(values)
            })
        
        return series
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics"""
        return {
            "metrics": {
                name: self.get_stats(name)
                for name in self.metrics.keys()
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges)
        }

# Global instance
metrics_aggregator = MetricsAggregator()
















