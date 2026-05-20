"""
Metrics Collector Service
=========================

Service for collecting and aggregating application metrics.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import json

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


@dataclass
class Metric:
    """Metric representation"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=DateTimeHelpers.now_utc)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricAggregation:
    """Metric aggregation result"""
    name: str
    type: MetricType
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float
    labels: Dict[str, str] = field(default_factory=dict)
    time_range: Dict[str, datetime] = field(default_factory=dict)


class MetricsCollector:
    """Metrics collector service"""
    
    def __init__(self, max_metrics_per_type: int = 10000, aggregation_interval: int = 60):
        self.max_metrics_per_type = max_metrics_per_type
        self.aggregation_interval = aggregation_interval
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_type))
        self.aggregations: Dict[str, MetricAggregation] = {}
        self.is_running = False
        self.aggregator_task: Optional[asyncio.Task] = None
        self.lock = threading.Lock()
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """Register default metrics"""
        # Application metrics
        self.register_counter("http_requests_total")
        self.register_counter("http_errors_total")
        self.register_histogram("http_request_duration_seconds")
        self.register_gauge("active_connections")
        self.register_gauge("memory_usage_bytes")
        self.register_gauge("cpu_usage_percent")
        
        # Business metrics
        self.register_counter("workflows_created_total")
        self.register_counter("workflows_completed_total")
        self.register_counter("nodes_created_total")
        self.register_histogram("workflow_processing_time_seconds")
        self.register_gauge("active_workflows")
        
        # System metrics
        self.register_counter("database_queries_total")
        self.register_histogram("database_query_duration_seconds")
        self.register_counter("cache_hits_total")
        self.register_counter("cache_misses_total")
        self.register_gauge("cache_hit_rate")
        
        logger.info("Default metrics registered")
    
    def register_counter(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Register a counter metric"""
        self._register_metric(name, MetricType.COUNTER, labels)
    
    def register_gauge(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Register a gauge metric"""
        self._register_metric(name, MetricType.GAUGE, labels)
    
    def register_histogram(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Register a histogram metric"""
        self._register_metric(name, MetricType.HISTOGRAM, labels)
    
    def register_summary(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Register a summary metric"""
        self._register_metric(name, MetricType.SUMMARY, labels)
    
    def register_timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Register a timer metric"""
        self._register_metric(name, MetricType.TIMER, labels)
    
    def _register_metric(self, name: str, metric_type: MetricType, labels: Optional[Dict[str, str]] = None):
        """Register a metric"""
        metric_key = self._get_metric_key(name, labels or {})
        if metric_key not in self.metrics:
            self.metrics[metric_key] = deque(maxlen=self.max_metrics_per_type)
            logger.debug(f"Registered metric: {name} ({metric_type.value})")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self._record_metric(name, MetricType.COUNTER, value, labels or {})
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        self._record_metric(name, MetricType.GAUGE, value, labels or {})
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a histogram metric"""
        self._record_metric(name, MetricType.HISTOGRAM, value, labels or {})
    
    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a summary metric"""
        self._record_metric(name, MetricType.SUMMARY, value, labels or {})
    
    def time_function(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator to time function execution"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        duration = time.time() - start_time
                        self.observe_histogram(name, duration, labels)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        duration = time.time() - start_time
                        self.observe_histogram(name, duration, labels)
                return sync_wrapper
        return decorator
    
    def _record_metric(self, name: str, metric_type: MetricType, value: float, labels: Dict[str, str]):
        """Record a metric"""
        metric_key = self._get_metric_key(name, labels)
        
        with self.lock:
            if metric_key not in self.metrics:
                self.metrics[metric_key] = deque(maxlen=self.max_metrics_per_type)
            
            metric = Metric(
                name=name,
                type=metric_type,
                value=value,
                labels=labels,
                timestamp=DateTimeHelpers.now_utc()
            )
            
            self.metrics[metric_key].append(metric)
    
    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Get metric key"""
        if not labels:
            return name
        
        sorted_labels = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
        return f"{name}{{{label_str}}}"
    
    async def start(self):
        """Start the metrics collector"""
        if self.is_running:
            return
        
        self.is_running = True
        self.aggregator_task = asyncio.create_task(self._aggregator())
        logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop the metrics collector"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.aggregator_task:
            self.aggregator_task.cancel()
            try:
                await self.aggregator_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics collector stopped")
    
    async def _aggregator(self):
        """Aggregate metrics periodically"""
        while self.is_running:
            try:
                await self._aggregate_metrics()
                await asyncio.sleep(self.aggregation_interval)
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
                await asyncio.sleep(self.aggregation_interval)
    
    async def _aggregate_metrics(self):
        """Aggregate all metrics"""
        with self.lock:
            for metric_key, metric_deque in self.metrics.items():
                if not metric_deque:
                    continue
                
                # Get metrics from the last aggregation interval
                cutoff_time = DateTimeHelpers.now_utc() - timedelta(seconds=self.aggregation_interval)
                recent_metrics = [m for m in metric_deque if m.timestamp >= cutoff_time]
                
                if not recent_metrics:
                    continue
                
                # Aggregate metrics
                aggregation = self._calculate_aggregation(metric_key, recent_metrics)
                self.aggregations[metric_key] = aggregation
    
    def _calculate_aggregation(self, metric_key: str, metrics: List[Metric]) -> MetricAggregation:
        """Calculate metric aggregation"""
        if not metrics:
            return MetricAggregation(
                name="",
                type=MetricType.COUNTER,
                count=0,
                sum=0.0,
                min=0.0,
                max=0.0,
                avg=0.0,
                p50=0.0,
                p95=0.0,
                p99=0.0
            )
        
        values = [m.value for m in metrics]
        values.sort()
        
        count = len(values)
        sum_val = sum(values)
        min_val = min(values)
        max_val = max(values)
        avg_val = sum_val / count if count > 0 else 0.0
        
        # Calculate percentiles
        p50 = self._calculate_percentile(values, 50)
        p95 = self._calculate_percentile(values, 95)
        p99 = self._calculate_percentile(values, 99)
        
        # Extract name and labels from metric key
        name, labels = self._parse_metric_key(metric_key)
        
        return MetricAggregation(
            name=name,
            type=metrics[0].type,
            count=count,
            sum=sum_val,
            min=min_val,
            max=max_val,
            avg=avg_val,
            p50=p50,
            p95=p95,
            p99=p99,
            labels=labels,
            time_range={
                "start": min(m.timestamp for m in metrics),
                "end": max(m.timestamp for m in metrics)
            }
        )
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        
        index = int((percentile / 100.0) * (len(values) - 1))
        return values[index]
    
    def _parse_metric_key(self, metric_key: str) -> tuple[str, Dict[str, str]]:
        """Parse metric key to extract name and labels"""
        if "{" not in metric_key:
            return metric_key, {}
        
        name = metric_key.split("{")[0]
        labels_str = metric_key.split("{")[1].rstrip("}")
        
        labels = {}
        if labels_str:
            for label_pair in labels_str.split(","):
                if "=" in label_pair:
                    key, value = label_pair.split("=", 1)
                    labels[key] = value
        
        return name, labels
    
    def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[MetricAggregation]:
        """Get metric aggregation"""
        metric_key = self._get_metric_key(name, labels or {})
        return self.aggregations.get(metric_key)
    
    def get_all_metrics(self) -> Dict[str, MetricAggregation]:
        """Get all metric aggregations"""
        return self.aggregations.copy()
    
    def get_metrics_by_type(self, metric_type: MetricType) -> Dict[str, MetricAggregation]:
        """Get metrics by type"""
        return {
            key: agg for key, agg in self.aggregations.items()
            if agg.type == metric_type
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        total_metrics = len(self.aggregations)
        metrics_by_type = defaultdict(int)
        
        for agg in self.aggregations.values():
            metrics_by_type[agg.type.value] += 1
        
        return {
            "total_metrics": total_metrics,
            "metrics_by_type": dict(metrics_by_type),
            "collection_interval": self.aggregation_interval,
            "max_metrics_per_type": self.max_metrics_per_type,
            "is_running": self.is_running,
            "timestamp": DateTimeHelpers.now_utc().isoformat()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        if format == "json":
            return json.dumps({
                "metrics": {
                    key: {
                        "name": agg.name,
                        "type": agg.type.value,
                        "count": agg.count,
                        "sum": agg.sum,
                        "min": agg.min,
                        "max": agg.max,
                        "avg": agg.avg,
                        "p50": agg.p50,
                        "p95": agg.p95,
                        "p99": agg.p99,
                        "labels": agg.labels,
                        "time_range": {
                            "start": agg.time_range["start"].isoformat(),
                            "end": agg.time_range["end"].isoformat()
                        }
                    }
                    for key, agg in self.aggregations.items()
                },
                "summary": self.get_metrics_summary()
            }, indent=2)
        
        elif format == "prometheus":
            return self._export_prometheus_format()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for key, agg in self.aggregations.items():
            # Add help and type comments
            lines.append(f"# HELP {agg.name} {agg.name} metric")
            lines.append(f"# TYPE {agg.name} {agg.type.value}")
            
            # Add metric values
            if agg.labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in agg.labels.items())
                lines.append(f"{agg.name}{{{label_str}}} {agg.avg}")
            else:
                lines.append(f"{agg.name} {agg.avg}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def clear_metrics(self):
        """Clear all metrics"""
        with self.lock:
            self.metrics.clear()
            self.aggregations.clear()
        logger.info("All metrics cleared")


# Global metrics collector
metrics_collector = MetricsCollector()


# Utility functions
async def start_metrics_collector():
    """Start the metrics collector"""
    await metrics_collector.start()


async def stop_metrics_collector():
    """Stop the metrics collector"""
    await metrics_collector.stop()


def increment_counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
    """Increment a counter metric"""
    metrics_collector.increment_counter(name, value, labels)


def set_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Set a gauge metric value"""
    metrics_collector.set_gauge(name, value, labels)


def observe_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Observe a histogram metric"""
    metrics_collector.observe_histogram(name, value, labels)


def observe_summary(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Observe a summary metric"""
    metrics_collector.observe_summary(name, value, labels)


def time_function(name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution"""
    return metrics_collector.time_function(name, labels)


def get_metric(name: str, labels: Optional[Dict[str, str]] = None) -> Optional[MetricAggregation]:
    """Get metric aggregation"""
    return metrics_collector.get_metric(name, labels)


def get_all_metrics() -> Dict[str, MetricAggregation]:
    """Get all metric aggregations"""
    return metrics_collector.get_all_metrics()


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary"""
    return metrics_collector.get_metrics_summary()


def export_metrics(format: str = "json") -> str:
    """Export metrics in specified format"""
    return metrics_collector.export_metrics(format)


def clear_metrics():
    """Clear all metrics"""
    metrics_collector.clear_metrics()


# Common metric recording functions
def record_http_request(method: str, endpoint: str, status_code: int, duration: float):
    """Record HTTP request metrics"""
    labels = {
        "method": method,
        "endpoint": endpoint,
        "status_code": str(status_code)
    }
    
    increment_counter("http_requests_total", 1.0, labels)
    observe_histogram("http_request_duration_seconds", duration, labels)
    
    if status_code >= 400:
        increment_counter("http_errors_total", 1.0, labels)


def record_workflow_operation(operation: str, duration: float, success: bool):
    """Record workflow operation metrics"""
    labels = {
        "operation": operation,
        "success": str(success).lower()
    }
    
    increment_counter("workflow_operations_total", 1.0, labels)
    observe_histogram("workflow_operation_duration_seconds", duration, labels)


def record_database_operation(operation: str, table: str, duration: float):
    """Record database operation metrics"""
    labels = {
        "operation": operation,
        "table": table
    }
    
    increment_counter("database_queries_total", 1.0, labels)
    observe_histogram("database_query_duration_seconds", duration, labels)


def record_cache_operation(operation: str, hit: bool):
    """Record cache operation metrics"""
    labels = {
        "operation": operation
    }
    
    if hit:
        increment_counter("cache_hits_total", 1.0, labels)
    else:
        increment_counter("cache_misses_total", 1.0, labels)
    
    # Update cache hit rate
    total_hits = metrics_collector.get_metric("cache_hits_total", labels)
    total_misses = metrics_collector.get_metric("cache_misses_total", labels)
    
    if total_hits and total_misses:
        hit_rate = total_hits.sum / (total_hits.sum + total_misses.sum)
        set_gauge("cache_hit_rate", hit_rate, labels)




