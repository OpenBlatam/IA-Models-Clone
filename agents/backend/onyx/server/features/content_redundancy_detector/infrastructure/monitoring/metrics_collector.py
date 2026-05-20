"""
Advanced Metrics Collection System - Real-time metrics and analytics
Production-ready metrics collection and aggregation
"""

import time
import asyncio
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import statistics
from datetime import datetime, timedelta

class MetricType(Enum):
    """Types of metrics supported"""
    COUNTER = "counter"  # Incremental values
    GAUGE = "gauge"      # Current values
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"      # Duration measurements
    RATE = "rate"        # Rate of change

@dataclass
class MetricData:
    """Individual metric data point"""
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class MetricSummary:
    """Summary statistics for a metric"""
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    p95: float
    p99: float
    std_dev: float
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """Advanced metrics collection and aggregation system"""
    
    def __init__(
        self,
        retention_period: int = 3600,  # 1 hour
        aggregation_interval: int = 60,  # 1 minute
        max_metrics_per_type: int = 10000
    ):
        self.retention_period = retention_period
        self.aggregation_interval = aggregation_interval
        self.max_metrics_per_type = max_metrics_per_type
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_type))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Aggregation
        self.aggregated_metrics: Dict[str, MetricSummary] = {}
        self.last_aggregation = time.time()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background tasks
        self.aggregation_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Callbacks
        self.callbacks: List[Callable[[MetricData], None]] = []

    async def start(self):
        """Start metrics collection and background tasks"""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.aggregation_task = asyncio.create_task(self._aggregation_worker())
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())

    async def stop(self):
        """Stop metrics collection and background tasks"""
        self.running = False
        
        if self.aggregation_task:
            self.aggregation_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            self.aggregation_task,
            self.cleanup_task,
            return_exceptions=True
        )

    def add_callback(self, callback: Callable[[MetricData], None]):
        """Add callback for metric events"""
        self.callbacks.append(callback)

    def _notify_callbacks(self, metric: MetricData):
        """Notify all callbacks of new metric"""
        for callback in self.callbacks:
            try:
                callback(metric)
            except Exception as e:
                print(f"Callback error: {e}")

    def counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Record a counter metric"""
        with self.lock:
            self.counters[name] += value
            
            metric = MetricData(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.COUNTER
            )
            
            self.metrics[name].append(metric)
            self._notify_callbacks(metric)

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric"""
        with self.lock:
            self.gauges[name] = value
            
            metric = MetricData(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.GAUGE
            )
            
            self.metrics[name].append(metric)
            self._notify_callbacks(metric)

    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric"""
        with self.lock:
            self.histograms[name].append(value)
            
            # Keep only recent values (last 1000)
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            
            metric = MetricData(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.HISTOGRAM
            )
            
            self.metrics[name].append(metric)
            self._notify_callbacks(metric)

    def timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer metric"""
        with self.lock:
            self.timers[name].append(duration)
            
            # Keep only recent values (last 1000)
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
            
            metric = MetricData(
                name=name,
                value=duration,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.TIMER
            )
            
            self.metrics[name].append(metric)
            self._notify_callbacks(metric)

    def rate(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a rate metric (events per second)"""
        with self.lock:
            metric = MetricData(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.RATE
            )
            
            self.metrics[name].append(metric)
            self._notify_callbacks(metric)

    def time_function(self, name: str, func: Callable, *args, tags: Dict[str, str] = None, **kwargs):
        """Time a function execution"""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            self.timer(name, duration, tags)

    async def time_async_function(self, name: str, func: Callable, *args, tags: Dict[str, str] = None, **kwargs):
        """Time an async function execution"""
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            self.timer(name, duration, tags)

    def get_metric(self, name: str, metric_type: MetricType = None) -> Union[float, List[float], None]:
        """Get current value of a metric"""
        with self.lock:
            if metric_type == MetricType.COUNTER:
                return self.counters.get(name, 0.0)
            elif metric_type == MetricType.GAUGE:
                return self.gauges.get(name, 0.0)
            elif metric_type == MetricType.HISTOGRAM:
                return self.histograms.get(name, [])
            elif metric_type == MetricType.TIMER:
                return self.timers.get(name, [])
            else:
                # Return most recent value
                if name in self.metrics and self.metrics[name]:
                    return self.metrics[name][-1].value
                return None

    def get_metric_summary(self, name: str, metric_type: MetricType = None) -> Optional[MetricSummary]:
        """Get summary statistics for a metric"""
        with self.lock:
            if name in self.aggregated_metrics:
                return self.aggregated_metrics[name]
            
            # Calculate on-demand if not aggregated
            values = []
            if metric_type == MetricType.HISTOGRAM:
                values = self.histograms.get(name, [])
            elif metric_type == MetricType.TIMER:
                values = self.timers.get(name, [])
            else:
                # Get values from metrics
                if name in self.metrics:
                    values = [m.value for m in self.metrics[name]]
            
            if not values:
                return None
            
            return self._calculate_summary(name, values, {})

    def _calculate_summary(self, name: str, values: List[float], tags: Dict[str, str]) -> MetricSummary:
        """Calculate summary statistics for values"""
        if not values:
            return MetricSummary(
                name=name,
                count=0, sum=0, min=0, max=0, mean=0, median=0, p95=0, p99=0, std_dev=0,
                tags=tags
            )
        
        sorted_values = sorted(values)
        count = len(values)
        total = sum(values)
        
        return MetricSummary(
            name=name,
            count=count,
            sum=total,
            min=min(values),
            max=max(values),
            mean=total / count,
            median=sorted_values[count // 2],
            p95=sorted_values[int(count * 0.95)] if count > 0 else 0,
            p99=sorted_values[int(count * 0.99)] if count > 0 else 0,
            std_dev=statistics.stdev(values) if count > 1 else 0,
            tags=tags
        )

    async def _aggregation_worker(self):
        """Background worker for metric aggregation"""
        while self.running:
            try:
                await asyncio.sleep(self.aggregation_interval)
                await self._aggregate_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Aggregation error: {e}")

    async def _cleanup_worker(self):
        """Background worker for metric cleanup"""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup error: {e}")

    async def _aggregate_metrics(self):
        """Aggregate metrics for summary statistics"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - self.aggregation_interval
            
            # Aggregate all metrics
            for name, metric_deque in self.metrics.items():
                if not metric_deque:
                    continue
                
                # Get recent values
                recent_values = [
                    m.value for m in metric_deque
                    if m.timestamp >= cutoff_time
                ]
                
                if recent_values:
                    # Calculate summary
                    summary = self._calculate_summary(name, recent_values, {})
                    self.aggregated_metrics[name] = summary
            
            self.last_aggregation = current_time

    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory leaks"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - self.retention_period
            
            # Clean up old metrics
            for name, metric_deque in self.metrics.items():
                # Remove old metrics
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()
            
            # Clean up old histograms and timers
            for name in list(self.histograms.keys()):
                if not self.histograms[name]:
                    del self.histograms[name]
            
            for name in list(self.timers.keys()):
                if not self.timers[name]:
                    del self.timers[name]

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {name: len(values) for name, values in self.histograms.items()},
                "timers": {name: len(values) for name, values in self.timers.items()},
                "aggregated": {name: {
                    "count": summary.count,
                    "mean": summary.mean,
                    "p95": summary.p95,
                    "p99": summary.p99
                } for name, summary in self.aggregated_metrics.items()},
                "last_aggregation": self.last_aggregation
            }

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        metrics_data = self.get_all_metrics()
        
        if format == "json":
            return json.dumps(metrics_data, indent=2)
        elif format == "prometheus":
            return self._export_prometheus_format()
        else:
            return str(metrics_data)

    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Histograms
        for name, values in self.histograms.items():
            if values:
                summary = self._calculate_summary(name, values, {})
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {summary.count}")
                lines.append(f"{name}_sum {summary.sum}")
                lines.append(f"{name}_mean {summary.mean}")
                lines.append(f"{name}_p95 {summary.p95}")
                lines.append(f"{name}_p99 {summary.p99}")
        
        return "\n".join(lines)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of metrics system"""
        with self.lock:
            total_metrics = sum(len(deque) for deque in self.metrics.values())
            memory_usage = sum(
                len(values) for values in self.histograms.values()
            ) + sum(
                len(values) for values in self.timers.values()
            )
            
            return {
                "status": "healthy" if self.running else "stopped",
                "total_metrics": total_metrics,
                "memory_usage": memory_usage,
                "aggregation_running": self.aggregation_task is not None,
                "cleanup_running": self.cleanup_task is not None,
                "last_aggregation": self.last_aggregation,
                "callbacks_registered": len(self.callbacks)
            }





