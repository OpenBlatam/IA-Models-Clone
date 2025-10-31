"""
Metrics Collector Implementation

High-performance metrics collection with multiple data types,
real-time aggregation, and efficient storage.
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import weakref
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric data types"""
    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"      # Current value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Quantiles and count
    TIMER = "timer"      # Duration measurements
    RATE = "rate"        # Rate of change


class AggregationType(Enum):
    """Aggregation types"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    STDDEV = "stddev"
    VARIANCE = "variance"


@dataclass
class MetricValue:
    """Metric value with metadata"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "type": self.metric_type.value,
            "unit": self.unit,
            "description": self.description
        }


@dataclass
class MetricSeries:
    """Time series of metric values"""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=10000))
    metric_type: MetricType = MetricType.GAUGE
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def add_value(self, value: Union[int, float], timestamp: Optional[datetime] = None):
        """Add value to series"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        metric_value = MetricValue(
            name=self.name,
            value=value,
            timestamp=timestamp,
            labels=self.labels,
            metric_type=self.metric_type,
            unit=self.unit,
            description=self.description
        )
        
        self.values.append(metric_value)
    
    def get_latest(self) -> Optional[MetricValue]:
        """Get latest value"""
        return self.values[-1] if self.values else None
    
    def get_values(self, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[MetricValue]:
        """Get values in time range"""
        if not start_time and not end_time:
            return list(self.values)
        
        filtered = []
        for value in self.values:
            if start_time and value.timestamp < start_time:
                continue
            if end_time and value.timestamp > end_time:
                continue
            filtered.append(value)
        
        return filtered
    
    def aggregate(self, aggregation: AggregationType, 
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None) -> Optional[float]:
        """Aggregate values"""
        values = self.get_values(start_time, end_time)
        if not values:
            return None
        
        numeric_values = [v.value for v in values]
        
        if aggregation == AggregationType.SUM:
            return sum(numeric_values)
        elif aggregation == AggregationType.AVG:
            return statistics.mean(numeric_values)
        elif aggregation == AggregationType.MIN:
            return min(numeric_values)
        elif aggregation == AggregationType.MAX:
            return max(numeric_values)
        elif aggregation == AggregationType.COUNT:
            return len(numeric_values)
        elif aggregation == AggregationType.MEDIAN:
            return statistics.median(numeric_values)
        elif aggregation == AggregationType.STDDEV:
            return statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0
        elif aggregation == AggregationType.VARIANCE:
            return statistics.variance(numeric_values) if len(numeric_values) > 1 else 0.0
        else:
            return None


class MetricsCollector:
    """High-performance metrics collector"""
    
    def __init__(self, max_series: int = 1000, max_values_per_series: int = 10000):
        self.max_series = max_series
        self.max_values_per_series = max_values_per_series
        self._series: Dict[str, MetricSeries] = {}
        self._lock = asyncio.Lock()
        self._listeners: List[Callable[[MetricValue], None]] = []
        self._aggregation_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def start(self):
        """Start metrics collector"""
        self._running = True
        logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop metrics collector"""
        self._running = False
        
        # Cancel aggregation tasks
        for task in self._aggregation_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._aggregation_tasks:
            await asyncio.gather(*self._aggregation_tasks.values(), return_exceptions=True)
        
        self._aggregation_tasks.clear()
        logger.info("Metrics collector stopped")
    
    def create_series(self, name: str, metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None,
                     unit: Optional[str] = None,
                     description: Optional[str] = None) -> MetricSeries:
        """Create a new metric series"""
        series_key = self._create_series_key(name, labels or {})
        
        if series_key in self._series:
            return self._series[series_key]
        
        if len(self._series) >= self.max_series:
            # Remove oldest series
            oldest_key = min(self._series.keys(), 
                           key=lambda k: self._series[k].values[0].timestamp if self._series[k].values else datetime.min)
            del self._series[oldest_key]
        
        series = MetricSeries(
            name=name,
            metric_type=metric_type,
            labels=labels or {},
            unit=unit,
            description=description
        )
        
        self._series[series_key] = series
        return series
    
    def _create_series_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create unique key for metric series"""
        if not labels:
            return name
        
        # Sort labels for consistent key generation
        sorted_labels = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
        return f"{name}{{{label_str}}}"
    
    async def record(self, name: str, value: Union[int, float],
                    labels: Optional[Dict[str, str]] = None,
                    metric_type: MetricType = MetricType.GAUGE,
                    unit: Optional[str] = None,
                    description: Optional[str] = None,
                    timestamp: Optional[datetime] = None) -> None:
        """Record a metric value"""
        series_key = self._create_series_key(name, labels or {})
        
        async with self._lock:
            if series_key not in self._series:
                self.create_series(name, metric_type, labels, unit, description)
            
            series = self._series[series_key]
            series.add_value(value, timestamp)
            
            # Create metric value for listeners
            metric_value = MetricValue(
                name=name,
                value=value,
                timestamp=timestamp or datetime.utcnow(),
                labels=labels or {},
                metric_type=metric_type,
                unit=unit,
                description=description
            )
            
            # Notify listeners
            for listener in self._listeners:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(metric_value)
                    else:
                        listener(metric_value)
                except Exception as e:
                    logger.error(f"Error in metrics listener: {e}")
    
    def counter(self, name: str, value: Union[int, float] = 1,
               labels: Optional[Dict[str, str]] = None,
               description: Optional[str] = None) -> None:
        """Record counter metric"""
        asyncio.create_task(self.record(
            name, value, labels, MetricType.COUNTER, description=description
        ))
    
    def gauge(self, name: str, value: Union[int, float],
             labels: Optional[Dict[str, str]] = None,
             unit: Optional[str] = None,
             description: Optional[str] = None) -> None:
        """Record gauge metric"""
        asyncio.create_task(self.record(
            name, value, labels, MetricType.GAUGE, unit, description
        ))
    
    def histogram(self, name: str, value: Union[int, float],
                 labels: Optional[Dict[str, str]] = None,
                 buckets: Optional[List[float]] = None,
                 description: Optional[str] = None) -> None:
        """Record histogram metric"""
        asyncio.create_task(self.record(
            name, value, labels, MetricType.HISTOGRAM, description=description
        ))
    
    def timer(self, name: str, duration: float,
             labels: Optional[Dict[str, str]] = None,
             description: Optional[str] = None) -> None:
        """Record timer metric"""
        asyncio.create_task(self.record(
            name, duration, labels, MetricType.TIMER, "seconds", description
        ))
    
    def rate(self, name: str, value: Union[int, float],
            labels: Optional[Dict[str, str]] = None,
            description: Optional[str] = None) -> None:
        """Record rate metric"""
        asyncio.create_task(self.record(
            name, value, labels, MetricType.RATE, description=description
        ))
    
    def get_series(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[MetricSeries]:
        """Get metric series"""
        series_key = self._create_series_key(name, labels or {})
        return self._series.get(series_key)
    
    def get_all_series(self) -> Dict[str, MetricSeries]:
        """Get all metric series"""
        return self._series.copy()
    
    def get_series_names(self) -> List[str]:
        """Get all series names"""
        return list(set(series.name for series in self._series.values()))
    
    def add_listener(self, listener: Callable[[MetricValue], None]) -> None:
        """Add metrics listener"""
        self._listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[MetricValue], None]) -> None:
        """Remove metrics listener"""
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    async def aggregate_series(self, name: str, aggregation: AggregationType,
                             labels: Optional[Dict[str, str]] = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Optional[float]:
        """Aggregate metric series"""
        series = self.get_series(name, labels)
        if not series:
            return None
        
        return series.aggregate(aggregation, start_time, end_time)
    
    async def get_aggregated_metrics(self, time_window: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """Get aggregated metrics for all series"""
        end_time = datetime.utcnow()
        start_time = end_time - time_window
        
        aggregated = {}
        
        for series_key, series in self._series.items():
            if not series.values:
                continue
            
            # Get values in time window
            window_values = series.get_values(start_time, end_time)
            if not window_values:
                continue
            
            numeric_values = [v.value for v in window_values]
            
            aggregated[series_key] = {
                "name": series.name,
                "labels": series.labels,
                "type": series.metric_type.value,
                "unit": series.unit,
                "count": len(numeric_values),
                "sum": sum(numeric_values),
                "avg": statistics.mean(numeric_values),
                "min": min(numeric_values),
                "max": max(numeric_values),
                "median": statistics.median(numeric_values),
                "stddev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
                "latest": numeric_values[-1] if numeric_values else None,
                "time_window": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            }
        
        return aggregated
    
    def clear_series(self, name: Optional[str] = None, labels: Optional[Dict[str, str]] = None) -> None:
        """Clear metric series"""
        if name:
            series_key = self._create_series_key(name, labels or {})
            if series_key in self._series:
                del self._series[series_key]
        else:
            self._series.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics"""
        total_values = sum(len(series.values) for series in self._series.values())
        
        return {
            "series_count": len(self._series),
            "total_values": total_values,
            "max_series": self.max_series,
            "max_values_per_series": self.max_values_per_series,
            "listeners_count": len(self._listeners),
            "running": self._running
        }


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str,
                 labels: Optional[Dict[str, str]] = None,
                 description: Optional[str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.timer(self.name, duration, self.labels, self.description)


class Counter:
    """Counter metric helper"""
    
    def __init__(self, collector: MetricsCollector, name: str,
                 labels: Optional[Dict[str, str]] = None,
                 description: Optional[str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.description = description
        self._value = 0
    
    def increment(self, value: Union[int, float] = 1) -> None:
        """Increment counter"""
        self._value += value
        self.collector.counter(self.name, self._value, self.labels, self.description)
    
    def reset(self) -> None:
        """Reset counter"""
        self._value = 0
        self.collector.counter(self.name, self._value, self.labels, self.description)
    
    def get_value(self) -> Union[int, float]:
        """Get current value"""
        return self._value


class Gauge:
    """Gauge metric helper"""
    
    def __init__(self, collector: MetricsCollector, name: str,
                 labels: Optional[Dict[str, str]] = None,
                 unit: Optional[str] = None,
                 description: Optional[str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.unit = unit
        self.description = description
    
    def set(self, value: Union[int, float]) -> None:
        """Set gauge value"""
        self.collector.gauge(self.name, value, self.labels, self.unit, self.description)
    
    def increment(self, value: Union[int, float] = 1) -> None:
        """Increment gauge"""
        # This would need to get current value first in a real implementation
        self.collector.gauge(self.name, value, self.labels, self.unit, self.description)
    
    def decrement(self, value: Union[int, float] = 1) -> None:
        """Decrement gauge"""
        # This would need to get current value first in a real implementation
        self.collector.gauge(self.name, -value, self.labels, self.unit, self.description)


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Convenience functions
def counter(name: str, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None):
    """Record counter metric"""
    metrics_collector.counter(name, value, labels)


def gauge(name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None, unit: Optional[str] = None):
    """Record gauge metric"""
    metrics_collector.gauge(name, value, labels, unit)


def timer(name: str, duration: float, labels: Optional[Dict[str, str]] = None):
    """Record timer metric"""
    metrics_collector.timer(name, duration, labels)


def create_timer(name: str, labels: Optional[Dict[str, str]] = None, description: Optional[str] = None) -> Timer:
    """Create timer context manager"""
    return Timer(metrics_collector, name, labels, description)


def create_counter(name: str, labels: Optional[Dict[str, str]] = None, description: Optional[str] = None) -> Counter:
    """Create counter helper"""
    return Counter(metrics_collector, name, labels, description)


def create_gauge(name: str, labels: Optional[Dict[str, str]] = None, unit: Optional[str] = None, description: Optional[str] = None) -> Gauge:
    """Create gauge helper"""
    return Gauge(metrics_collector, name, labels, unit, description)





















