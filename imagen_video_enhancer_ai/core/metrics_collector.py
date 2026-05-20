"""
Metrics Collector for Imagen Video Enhancer AI
=============================================

Advanced metrics collection and analysis.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "tags": self.tags
        }


class MetricsCollector:
    """
    Collects and analyzes metrics.
    
    Features:
    - Time-series metrics
    - Aggregations
    - Percentiles
    - Rate calculations
    - Tag-based filtering
    """
    
    def __init__(self, max_points: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_points: Maximum number of points per metric
        """
        self.max_points = max_points
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
    
    def record(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a metric point.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            tags: Optional tags
        """
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        
        self._metrics[metric_name].append(point)
        logger.debug(f"Recorded metric {metric_name}: {value}")
    
    def increment(self, counter_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """
        Increment a counter.
        
        Args:
            counter_name: Counter name
            value: Increment value
            tags: Optional tags
        """
        self._counters[counter_name] += value
        self.record(f"counter.{counter_name}", self._counters[counter_name], tags)
    
    def set_gauge(self, gauge_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Set a gauge value.
        
        Args:
            gauge_name: Gauge name
            value: Gauge value
            tags: Optional tags
        """
        self._gauges[gauge_name] = value
        self.record(f"gauge.{gauge_name}", value, tags)
    
    def record_histogram(
        self,
        histogram_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a histogram value.
        
        Args:
            histogram_name: Histogram name
            value: Value to record
            tags: Optional tags
        """
        self._histograms[histogram_name].append(value)
        # Keep only last N values
        if len(self._histograms[histogram_name]) > 1000:
            self._histograms[histogram_name] = self._histograms[histogram_name][-1000:]
        
        self.record(f"histogram.{histogram_name}", value, tags)
    
    def get_metric(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[MetricPoint]:
        """
        Get metric points.
        
        Args:
            metric_name: Metric name
            start_time: Optional start time filter
            end_time: Optional end time filter
            tags: Optional tag filter
            
        Returns:
            List of metric points
        """
        if metric_name not in self._metrics:
            return []
        
        points = list(self._metrics[metric_name])
        
        # Filter by time
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        # Filter by tags
        if tags:
            points = [
                p for p in points
                if all(p.tags.get(k) == v for k, v in tags.items())
            ]
        
        return points
    
    def get_statistics(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Metric name
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            Statistics dictionary
        """
        points = self.get_metric(metric_name, start_time, end_time)
        
        if not points:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "sum": None
            }
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
            "first": values[0] if values else None,
            "last": values[-1] if values else None
        }
    
    def get_percentiles(
        self,
        histogram_name: str,
        percentiles: List[float] = [50, 75, 90, 95, 99]
    ) -> Dict[str, float]:
        """
        Get percentiles for a histogram.
        
        Args:
            histogram_name: Histogram name
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary of percentile values
        """
        if histogram_name not in self._histograms:
            return {}
        
        values = sorted(self._histograms[histogram_name])
        if not values:
            return {}
        
        result = {}
        for p in percentiles:
            index = int(len(values) * p / 100)
            index = min(index, len(values) - 1)
            result[f"p{p}"] = values[index]
        
        return result
    
    def get_rate(
        self,
        metric_name: str,
        window_seconds: float = 60.0
    ) -> float:
        """
        Calculate rate (events per second) for a metric.
        
        Args:
            metric_name: Metric name
            window_seconds: Time window in seconds
            
        Returns:
            Rate in events per second
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=window_seconds)
        
        points = self.get_metric(metric_name, start_time, end_time)
        
        if not points:
            return 0.0
        
        time_span = (points[-1].timestamp - points[0].timestamp).total_seconds()
        if time_span == 0:
            return 0.0
        
        return len(points) / time_span
    
    def get_all_metrics(self) -> List[str]:
        """Get list of all metric names."""
        return list(self._metrics.keys())
    
    def get_counters(self) -> Dict[str, int]:
        """Get all counter values."""
        return dict(self._counters)
    
    def get_gauges(self) -> Dict[str, float]:
        """Get all gauge values."""
        return dict(self._gauges)
    
    def reset(self, metric_name: Optional[str] = None):
        """
        Reset metrics.
        
        Args:
            metric_name: Optional specific metric to reset
        """
        if metric_name:
            if metric_name in self._metrics:
                self._metrics[metric_name].clear()
            if metric_name in self._counters:
                del self._counters[metric_name]
            if metric_name in self._gauges:
                del self._gauges[metric_name]
        else:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()




