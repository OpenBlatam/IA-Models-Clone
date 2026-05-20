"""
Advanced metrics system for KV cache.

This module provides comprehensive metrics collection, aggregation,
and analysis capabilities.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics


class MetricAggregation(Enum):
    """Metric aggregation methods."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"


@dataclass
class MetricDataPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric data."""
    metric_name: str
    aggregation: MetricAggregation
    value: float
    timestamp: float
    count: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    percentile_95: Optional[float] = None
    percentile_99: Optional[float] = None


class AdvancedMetricsCollector:
    """Advanced metrics collector."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._lock = threading.Lock()
        
    def record(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value."""
        data_point = MetricDataPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        
        with self._lock:
            self._metrics[metric_name].append(data_point)
            
    def get_metric(
        self,
        metric_name: str,
        window_seconds: Optional[float] = None
    ) -> List[MetricDataPoint]:
        """Get metric data points."""
        with self._lock:
            points = list(self._metrics.get(metric_name, []))
            
        if window_seconds:
            cutoff_time = time.time() - window_seconds
            points = [p for p in points if p.timestamp >= cutoff_time]
            
        return points
        
    def aggregate(
        self,
        metric_name: str,
        aggregation: MetricAggregation,
        window_seconds: Optional[float] = None,
        percentile: Optional[float] = None
    ) -> AggregatedMetric:
        """Aggregate metric data."""
        points = self.get_metric(metric_name, window_seconds)
        
        if not points:
            return AggregatedMetric(
                metric_name=metric_name,
                aggregation=aggregation,
                value=0.0,
                timestamp=time.time(),
                count=0
            )
            
        values = [p.value for p in points]
        
        if aggregation == MetricAggregation.SUM:
            agg_value = sum(values)
        elif aggregation == MetricAggregation.AVG:
            agg_value = statistics.mean(values)
        elif aggregation == MetricAggregation.MIN:
            agg_value = min(values)
        elif aggregation == MetricAggregation.MAX:
            agg_value = max(values)
        elif aggregation == MetricAggregation.COUNT:
            agg_value = len(values)
        elif aggregation == MetricAggregation.PERCENTILE:
            if percentile:
                agg_value = self._percentile(values, percentile)
            else:
                agg_value = statistics.median(values)
        else:
            agg_value = statistics.mean(values)
            
        # Calculate percentiles
        sorted_values = sorted(values)
        percentile_95 = self._percentile(sorted_values, 0.95) if len(sorted_values) > 1 else None
        percentile_99 = self._percentile(sorted_values, 0.99) if len(sorted_values) > 1 else None
        
        return AggregatedMetric(
            metric_name=metric_name,
            aggregation=aggregation,
            value=agg_value,
            timestamp=time.time(),
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            percentile_95=percentile_95,
            percentile_99=percentile_99
        )
        
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        k = (len(values) - 1) * percentile
        f = int(k)
        c = k - f
        
        if f + 1 < len(values):
            return values[f] + c * (values[f + 1] - values[f])
        return values[f]
        
    def get_all_metrics(self) -> List[str]:
        """Get list of all metric names."""
        with self._lock:
            return list(self._metrics.keys())
            
    def get_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metrics."""
        summary = {}
        
        for metric_name in self.get_all_metrics():
            avg_metric = self.aggregate(metric_name, MetricAggregation.AVG, window_seconds=3600)
            summary[metric_name] = {
                'avg': avg_metric.value,
                'min': avg_metric.min_value,
                'max': avg_metric.max_value,
                'count': avg_metric.count,
                'p95': avg_metric.percentile_95,
                'p99': avg_metric.percentile_99
            }
            
        return summary


class CacheMetricsAdvanced:
    """Advanced metrics for cache."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self.collector = AdvancedMetricsCollector(cache)
        self._auto_collect = True
        
        if self._auto_collect:
            self._setup_auto_collection()
            
    def _setup_auto_collection(self) -> None:
        """Setup automatic metrics collection."""
        # This would hook into cache operations
        # For now, it's a placeholder
        pass
        
    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True
    ) -> None:
        """Record cache operation."""
        self.collector.record(f"operation_{operation}_duration", duration)
        self.collector.record(f"operation_{operation}_count", 1.0)
        self.collector.record(f"operation_{operation}_success", 1.0 if success else 0.0)
        
    def get_operation_metrics(
        self,
        operation: str,
        window_seconds: float = 3600.0
    ) -> Dict[str, Any]:
        """Get metrics for a specific operation."""
        duration_avg = self.collector.aggregate(
            f"operation_{operation}_duration",
            MetricAggregation.AVG,
            window_seconds
        )
        
        count_total = self.collector.aggregate(
            f"operation_{operation}_count",
            MetricAggregation.SUM,
            window_seconds
        )
        
        success_rate = self.collector.aggregate(
            f"operation_{operation}_success",
            MetricAggregation.AVG,
            window_seconds
        )
        
        return {
            'avg_duration': duration_avg.value,
            'p95_duration': duration_avg.percentile_95,
            'p99_duration': duration_avg.percentile_99,
            'total_count': count_total.value,
            'success_rate': success_rate.value,
            'throughput': count_total.value / window_seconds if window_seconds > 0 else 0
        }



