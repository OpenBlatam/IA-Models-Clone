"""
Real-time Metrics System
=========================

Real-time metrics collection and aggregation system.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Metric time series."""
    name: str
    type: MetricType
    points: deque = field(default_factory=lambda: deque(maxlen=10000))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, timestamp: Optional[datetime] = None, labels: Optional[Dict[str, str]] = None):
        """
        Add metric point.
        
        Args:
            value: Metric value
            timestamp: Optional timestamp
            labels: Optional labels
        """
        point = MetricPoint(
            name=self.name,
            value=value,
            timestamp=timestamp or datetime.now(),
            tags=self.tags,
            labels=labels or {}
        )
        self.points.append(point)
    
    def get_recent(self, since: Optional[datetime] = None, limit: Optional[int] = None) -> List[MetricPoint]:
        """
        Get recent metric points.
        
        Args:
            since: Optional filter by timestamp
            limit: Optional limit results
            
        Returns:
            List of metric points
        """
        points = list(self.points)
        
        if since:
            points = [p for p in points if p.timestamp >= since]
        
        if limit:
            points = points[-limit:]
        
        return points
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get metric statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.points:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "sum": None
            }
        
        values = [p.value for p in self.points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values)
        }


class RealTimeMetricsCollector:
    """Real-time metrics collector."""
    
    def __init__(self):
        """Initialize real-time metrics collector."""
        self.metrics: Dict[str, MetricSeries] = {}
        self.lock = asyncio.Lock()
        self.aggregators: Dict[str, Callable] = {}
    
    async def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Record metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Metric type
            tags: Optional tags
            labels: Optional labels
        """
        async with self.lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name,
                    type=metric_type,
                    tags=tags or {}
                )
            
            self.metrics[name].add_point(value, labels=labels)
    
    async def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """
        Increment counter metric.
        
        Args:
            name: Metric name
            value: Increment value
            tags: Optional tags
        """
        await self.record(name, value, MetricType.COUNTER, tags=tags)
    
    async def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Set gauge metric.
        
        Args:
            name: Metric name
            value: Gauge value
            tags: Optional tags
        """
        await self.record(name, value, MetricType.GAUGE, tags=tags)
    
    async def observe_histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Observe histogram metric.
        
        Args:
            name: Metric name
            value: Observed value
            tags: Optional tags
            labels: Optional labels
        """
        await self.record(name, value, MetricType.HISTOGRAM, tags=tags, labels=labels)
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """
        Get metric series.
        
        Args:
            name: Metric name
            
        Returns:
            Metric series or None
        """
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """
        Get all metrics.
        
        Returns:
            Dictionary of name -> metric series
        """
        return self.metrics.copy()
    
    def get_metric_statistics(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metric statistics.
        
        Args:
            name: Metric name
            
        Returns:
            Statistics dictionary or None
        """
        metric = self.metrics.get(name)
        if not metric:
            return None
        
        stats = metric.get_statistics()
        stats["name"] = name
        stats["type"] = metric.type.value
        stats["tags"] = metric.tags
        
        return stats
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all metrics.
        
        Returns:
            Dictionary of name -> statistics
        """
        return {
            name: self.get_metric_statistics(name)
            for name in self.metrics.keys()
        }
    
    def clear_metric(self, name: str):
        """
        Clear metric.
        
        Args:
            name: Metric name
        """
        if name in self.metrics:
            del self.metrics[name]
    
    def clear_all(self):
        """Clear all metrics."""
        self.metrics.clear()



