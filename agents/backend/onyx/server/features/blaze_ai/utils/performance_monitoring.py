"""
Advanced performance monitoring and profiling for Blaze AI.

This module provides comprehensive performance monitoring, profiling,
metrics collection, and real-time analysis capabilities.
"""

import asyncio
import time
import psutil
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from collections import defaultdict, deque
import json
import statistics
from contextlib import asynccontextmanager, contextmanager
import tracemalloc
import cProfile
import pstats
import io

# =============================================================================
# Types
# =============================================================================

MetricValue = Union[int, float, str, bool]
MetricCallback = Callable[[str, MetricValue], None]

# =============================================================================
# Enums
# =============================================================================

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class ProfilingLevel(Enum):
    """Profiling levels."""
    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"
    FULL = "full"

# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    enable_monitoring: bool = True
    enable_profiling: bool = False
    enable_memory_tracking: bool = True
    enable_system_metrics: bool = True
    enable_custom_metrics: bool = True
    metrics_interval: float = 1.0  # seconds
    profiling_level: ProfilingLevel = ProfilingLevel.BASIC
    max_metrics_history: int = 1000
    enable_alerting: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class MetricConfig:
    """Configuration for individual metrics."""
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    aggregation: str = "last"  # last, min, max, avg, sum
    retention_period: float = 3600.0  # seconds

# =============================================================================
# Base Metric Classes
# =============================================================================

class BaseMetric(ABC):
    """Abstract base class for metrics."""
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.values: deque = deque(maxlen=1000)
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    @abstractmethod
    def update(self, value: MetricValue, labels: Optional[Dict[str, str]] = None):
        """Update metric value."""
        pass
    
    @abstractmethod
    def get_value(self, aggregation: Optional[str] = None) -> MetricValue:
        """Get metric value with optional aggregation."""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metric summary."""
        with self._lock:
            return {
                "name": self.config.name,
                "type": self.config.metric_type.value,
                "description": self.config.description,
                "unit": self.config.unit,
                "last_update": self.last_update,
                "value_count": len(self.values),
                "current_value": self.get_value(),
                "aggregated_values": {
                    "min": self.get_value("min"),
                    "max": self.get_value("max"),
                    "avg": self.get_value("avg"),
                    "sum": self.get_value("sum")
                }
            }

class CounterMetric(BaseMetric):
    """Counter metric implementation."""
    
    def __init__(self, config: MetricConfig):
        super().__init__(config)
        self.total = 0
    
    def update(self, value: MetricValue, labels: Optional[Dict[str, str]] = None):
        """Update counter value."""
        if isinstance(value, (int, float)):
            with self._lock:
                self.total += value
                self.values.append({
                    "value": value,
                    "timestamp": time.time(),
                    "labels": labels or {}
                })
                self.last_update = time.time()
    
    def get_value(self, aggregation: Optional[str] = None) -> MetricValue:
        """Get counter value."""
        with self._lock:
            if aggregation == "sum":
                return self.total
            elif aggregation == "last" and self.values:
                return self.values[-1]["value"]
            return self.total

class GaugeMetric(BaseMetric):
    """Gauge metric implementation."""
    
    def update(self, value: MetricValue, labels: Optional[Dict[str, str]] = None):
        """Update gauge value."""
        with self._lock:
            self.values.append({
                "value": value,
                "timestamp": time.time(),
                "labels": labels or {}
            })
            self.last_update = time.time()
    
    def get_value(self, aggregation: Optional[str] = None) -> MetricValue:
        """Get gauge value."""
        with self._lock:
            if not self.values:
                return 0
            
            if aggregation == "min":
                return min(v["value"] for v in self.values if isinstance(v["value"], (int, float)))
            elif aggregation == "max":
                return max(v["value"] for v in self.values if isinstance(v["value"], (int, float)))
            elif aggregation == "avg":
                numeric_values = [v["value"] for v in self.values if isinstance(v["value"], (int, float))]
                return statistics.mean(numeric_values) if numeric_values else 0
            else:  # last
                return self.values[-1]["value"]

class HistogramMetric(BaseMetric):
    """Histogram metric implementation."""
    
    def __init__(self, config: MetricConfig):
        super().__init__(config)
        self.buckets: Dict[str, int] = defaultdict(int)
        self.bucket_boundaries = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')]
    
    def update(self, value: MetricValue, labels: Optional[Dict[str, str]] = None):
        """Update histogram value."""
        if isinstance(value, (int, float)):
            with self._lock:
                # Find appropriate bucket
                bucket = "inf"
                for boundary in self.bucket_boundaries:
                    if value <= boundary:
                        bucket = str(boundary)
                        break
                
                self.buckets[bucket] += 1
                self.values.append({
                    "value": value,
                    "timestamp": time.time(),
                    "labels": labels or {},
                    "bucket": bucket
                })
                self.last_update = time.time()
    
    def get_value(self, aggregation: Optional[str] = None) -> Dict[str, Any]:
        """Get histogram value."""
        with self._lock:
            if not self.values:
                return {"buckets": self.buckets, "count": 0, "sum": 0}
            
            numeric_values = [v["value"] for v in self.values if isinstance(v["value"], (int, float))]
            return {
                "buckets": dict(self.buckets),
                "count": len(numeric_values),
                "sum": sum(numeric_values),
                "min": min(numeric_values) if numeric_values else 0,
                "max": max(numeric_values) if numeric_values else 0,
                "avg": statistics.mean(numeric_values) if numeric_values else 0
            }

# =============================================================================
# Performance Profiler
# =============================================================================

class PerformanceProfiler:
    """Performance profiling and analysis."""
    
    def __init__(self, level: ProfilingLevel = ProfilingLevel.BASIC):
        self.level = level
        self.profiler = None
        self.memory_snapshot = None
        self.active_profiles: Dict[str, Any] = {}
    
    @contextmanager
    def profile_function(self, name: str):
        """Profile a function execution."""
        if self.level == ProfilingLevel.NONE:
            yield
            return
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss if self.level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL] else 0
        
        if self.level == ProfilingLevel.FULL:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss if self.level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL] else 0
            
            if self.profiler:
                self.profiler.disable()
                stats = pstats.Stats(self.profiler)
                stats.sort_stats('cumulative')
                
                # Capture stats output
                output = io.StringIO()
                stats.print_stats(20, output)  # Top 20 functions
                stats_output = output.getvalue()
                output.close()
            
            profile_data = {
                "name": name,
                "execution_time": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "start_time": start_time,
                "end_time": end_time
            }
            
            if self.level == ProfilingLevel.FULL:
                profile_data["stats"] = stats_output
            
            self.active_profiles[name] = profile_data
    
    @asynccontextmanager
    async def profile_async_function(self, name: str):
        """Profile an async function execution."""
        if self.level == ProfilingLevel.NONE:
            yield
            return
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss if self.level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL] else 0
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss if self.level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL] else 0
            
            profile_data = {
                "name": name,
                "execution_time": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "start_time": start_time,
                "end_time": end_time,
                "async": True
            }
            
            self.active_profiles[name] = profile_data
    
    def start_memory_tracking(self):
        """Start memory tracking."""
        if self.level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL]:
            tracemalloc.start()
            self.memory_snapshot = tracemalloc.take_snapshot()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.memory_snapshot:
            return {}
        
        current_snapshot = tracemalloc.take_snapshot()
        stats = current_snapshot.compare_to(self.memory_snapshot, 'lineno')
        
        return {
            "current_memory": psutil.Process().memory_info().rss,
            "memory_growth": sum(stat.size_diff for stat in stats),
            "top_allocations": [
                {
                    "file": stat.traceback.format()[-1],
                    "size_diff": stat.size_diff,
                    "count_diff": stat.count_diff
                }
                for stat in stats[:10]
            ]
        }
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.active_profiles:
            return {}
        
        total_execution_time = sum(p["execution_time"] for p in self.active_profiles.values())
        total_memory_delta = sum(p["memory_delta"] for p in self.active_profiles.values())
        
        return {
            "total_profiles": len(self.active_profiles),
            "total_execution_time": total_execution_time,
            "total_memory_delta": total_memory_delta,
            "profiles": self.active_profiles,
            "memory_stats": self.get_memory_stats()
        }

# =============================================================================
# System Metrics Collector
# =============================================================================

class SystemMetricsCollector:
    """Collect system-level metrics."""
    
    def __init__(self):
        self.last_collection = time.time()
        self.collection_interval = 1.0  # seconds
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        current_time = time.time()
        
        # Only collect if enough time has passed
        if current_time - self.last_collection < self.collection_interval:
            return {}
        
        self.last_collection = current_time
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return {
                "timestamp": current_time,
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            }
        except Exception as e:
            return {"error": str(e), "timestamp": current_time}

# =============================================================================
# Performance Monitor
# =============================================================================

class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics: Dict[str, BaseMetric] = {}
        self.profiler = PerformanceProfiler(config.profiling_level)
        self.system_collector = SystemMetricsCollector()
        self.metric_callbacks: List[MetricCallback] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        if config.enable_monitoring:
            self._setup_default_metrics()
    
    def _setup_default_metrics(self):
        """Setup default system metrics."""
        default_metrics = [
            MetricConfig("system.cpu.percent", MetricType.GAUGE, "CPU usage percentage", "%"),
            MetricConfig("system.memory.percent", MetricType.GAUGE, "Memory usage percentage", "%"),
            MetricConfig("system.disk.percent", MetricType.GAUGE, "Disk usage percentage", "%"),
            MetricConfig("requests.total", MetricType.COUNTER, "Total requests", "count"),
            MetricConfig("requests.duration", MetricType.HISTOGRAM, "Request duration", "seconds"),
            MetricConfig("errors.total", MetricType.COUNTER, "Total errors", "count")
        ]
        
        for metric_config in default_metrics:
            self.add_metric(metric_config)
    
    def add_metric(self, config: MetricConfig) -> BaseMetric:
        """Add a new metric."""
        with self._lock:
            if config.metric_type == MetricType.COUNTER:
                metric = CounterMetric(config)
            elif config.metric_type == MetricType.GAUGE:
                metric = GaugeMetric(config)
            elif config.metric_type == MetricType.HISTOGRAM:
                metric = HistogramMetric(config)
            else:
                raise ValueError(f"Unsupported metric type: {config.metric_type}")
            
            self.metrics[config.name] = metric
            return metric
    
    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    def update_metric(self, name: str, value: MetricValue, labels: Optional[Dict[str, str]] = None):
        """Update a metric value."""
        metric = self.metrics.get(name)
        if metric:
            metric.update(value, labels)
            
            # Notify callbacks
            for callback in self.metric_callbacks:
                try:
                    callback(name, value)
                except Exception:
                    pass  # Ignore callback errors
    
    def add_metric_callback(self, callback: MetricCallback):
        """Add a callback for metric updates."""
        self.metric_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_task and not self.monitoring_task.done():
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                if self.config.enable_system_metrics:
                    system_metrics = self.system_collector.collect_system_metrics()
                    if system_metrics:
                        self.update_metric("system.cpu.percent", system_metrics.get("cpu", {}).get("percent", 0))
                        self.update_metric("system.memory.percent", system_metrics.get("memory", {}).get("percent", 0))
                        self.update_metric("system.disk.percent", system_metrics.get("disk", {}).get("percent", 0))
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                print(f"Monitoring error: {e}")
                await asyncio.sleep(self.config.metrics_interval)
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            summary = {
                "timestamp": time.time(),
                "metrics_count": len(self.metrics),
                "metrics": {}
            }
            
            for name, metric in self.metrics.items():
                summary["metrics"][name] = metric.get_summary()
            
            if self.config.enable_profiling:
                summary["profiling"] = self.profiler.get_profile_summary()
            
            return summary
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        summary = self.get_metrics_summary()
        
        if format.lower() == "json":
            return json.dumps(summary, indent=2, default=str)
        elif format.lower() == "prometheus":
            return self._export_prometheus(summary)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus(self, summary: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, metric_data in summary.get("metrics", {}).items():
            metric_name = metric_name.replace(".", "_")
            current_value = metric_data.get("current_value", 0)
            
            if isinstance(current_value, (int, float)):
                lines.append(f"{metric_name} {current_value}")
        
        return "\n".join(lines)

# =============================================================================
# Decorators
# =============================================================================

def monitor_performance(metric_name: str, metric_type: MetricType = MetricType.GAUGE):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Update metrics if monitor is available
                try:
                    from . import get_performance_monitor
                    monitor = get_performance_monitor()
                    if monitor:
                        monitor.update_metric(metric_name, execution_time)
                except ImportError:
                    pass
                
                return result
            except Exception as e:
                # Record error metric
                try:
                    from . import get_performance_monitor
                    monitor = get_performance_monitor()
                    if monitor:
                        monitor.update_metric("errors.total", 1)
                except ImportError:
                    pass
                raise
        
        return wrapper
    return decorator

def profile_function(level: ProfilingLevel = ProfilingLevel.BASIC):
    """Decorator to profile function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                from . import get_performance_monitor
                monitor = get_performance_monitor()
                if monitor and monitor.config.enable_profiling:
                    with monitor.profiler.profile_function(func.__name__):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except ImportError:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# =============================================================================
# Global Monitor Instance
# =============================================================================

_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """Get global performance monitor instance."""
    return _performance_monitor

def set_performance_monitor(monitor: PerformanceMonitor):
    """Set global performance monitor instance."""
    global _performance_monitor
    _performance_monitor = monitor
