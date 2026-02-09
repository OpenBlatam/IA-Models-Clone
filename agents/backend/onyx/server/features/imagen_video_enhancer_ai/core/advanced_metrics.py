"""
Advanced Metrics System
========================

Advanced metrics collection and aggregation system.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric type."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "type": self.metric_type.value
        }


@dataclass
class MetricAggregation:
    """Metric aggregation result."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    avg: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "avg": self.avg,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


class AdvancedMetricsCollector:
    """Advanced metrics collector with aggregation."""
    
    def __init__(self, retention_hours: float = 24.0):
        """
        Initialize metrics collector.
        
        Args:
            retention_hours: Hours to retain metrics
        """
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Increment a counter.
        
        Args:
            name: Metric name
            value: Increment value
            tags: Optional tags
        """
        key = self._make_key(name, tags)
        self.counters[key] += value
        
        point = MetricPoint(
            name=name,
            value=self.counters[key],
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=MetricType.COUNTER
        )
        self.metrics[key].append(point)
    
    def set_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Set a gauge value.
        
        Args:
            name: Metric name
            value: Gauge value
            tags: Optional tags
        """
        key = self._make_key(name, tags)
        self.gauges[key] = value
        
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=MetricType.GAUGE
        )
        self.metrics[key].append(point)
    
    def record_histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a histogram value.
        
        Args:
            name: Metric name
            value: Histogram value
            tags: Optional tags
        """
        key = self._make_key(name, tags)
        self.histograms[key].append(value)
        
        # Limit histogram size
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=MetricType.HISTOGRAM
        )
        self.metrics[key].append(point)
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Make metric key from name and tags."""
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}:{tag_str}"
        return name
    
    def get_metric(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """
        Get current metric value.
        
        Args:
            name: Metric name
            tags: Optional tags
            
        Returns:
            Metric value or None
        """
        key = self._make_key(name, tags)
        
        if key in self.counters:
            return self.counters[key]
        if key in self.gauges:
            return self.gauges[key]
        return None
    
    def aggregate(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        window_minutes: float = 60.0
    ) -> Optional[MetricAggregation]:
        """
        Aggregate metrics over a time window.
        
        Args:
            name: Metric name
            tags: Optional tags
            window_minutes: Time window in minutes
            
        Returns:
            Aggregation result or None
        """
        key = self._make_key(name, tags)
        
        if key not in self.metrics:
            return None
        
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        points = [
            point for point in self.metrics[key]
            if point.timestamp >= cutoff
        ]
        
        if not points:
            return None
        
        values = [point.value for point in points]
        
        return MetricAggregation(
            name=name,
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            avg=sum(values) / len(values),
            timestamp=datetime.now(),
            tags=tags or {}
        )
    
    def get_percentile(
        self,
        name: str,
        percentile: float,
        tags: Optional[Dict[str, str]] = None,
        window_minutes: float = 60.0
    ) -> Optional[float]:
        """
        Get percentile value.
        
        Args:
            name: Metric name
            percentile: Percentile (0-100)
            tags: Optional tags
            window_minutes: Time window in minutes
            
        Returns:
            Percentile value or None
        """
        key = self._make_key(name, tags)
        
        if key not in self.histograms:
            return None
        
        values = self.histograms[key]
        if not values:
            return None
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histogram_counts": {
                key: len(values) for key, values in self.histograms.items()
            }
        }




