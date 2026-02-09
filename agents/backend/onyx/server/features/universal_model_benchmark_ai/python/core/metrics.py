"""
Metrics Module - Enhanced Prometheus metrics collection.

Provides:
- Comprehensive Prometheus metrics
- Custom metrics with labels
- Metrics aggregation and export
- Performance metrics tracking
- System resource metrics
- Error tracking and categorization
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, generate_latest,
        CollectorRegistry, REGISTRY, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    Gauge = None
    Summary = None
    generate_latest = None
    CollectorRegistry = None
    REGISTRY = None
    start_http_server = None

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Metric value with timestamp and labels."""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }


@dataclass
class MetricDefinition:
    """Metric definition."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


class MetricsCollector:
    """
    Enhanced Prometheus metrics collector.
    
    Features:
    - Comprehensive benchmark metrics
    - System resource metrics
    - Error tracking
    - API metrics
    - Custom metrics support
    """
    
    def __init__(self, registry: Optional[Any] = None):
        """
        Initialize metrics collector.
        
        Args:
            registry: Optional Prometheus registry
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Metrics will be disabled.")
            self.enabled = False
            self.registry = None
            return
        
        self.enabled = True
        self.registry = registry or REGISTRY
        
        # Benchmark metrics
        self.benchmark_requests = Counter(
            'benchmark_requests_total',
            'Total benchmark requests',
            ['model', 'benchmark', 'status']
        )
        
        self.benchmark_duration = Histogram(
            'benchmark_duration_seconds',
            'Benchmark execution duration',
            ['model', 'benchmark'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
        )
        
        self.benchmark_accuracy = Gauge(
            'benchmark_accuracy',
            'Benchmark accuracy score',
            ['model', 'benchmark']
        )
        
        self.benchmark_throughput = Gauge(
            'benchmark_throughput_tokens_per_second',
            'Benchmark throughput in tokens per second',
            ['model', 'benchmark']
        )
        
        self.benchmark_latency = Histogram(
            'benchmark_latency_seconds',
            'Benchmark latency distribution',
            ['model', 'benchmark', 'percentile'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.benchmark_samples = Counter(
            'benchmark_samples_total',
            'Total benchmark samples processed',
            ['model', 'benchmark', 'result']  # result: correct, incorrect
        )
        
        # System metrics
        self.active_experiments = Gauge(
            'active_experiments',
            'Number of active experiments'
        )
        
        self.total_models = Gauge(
            'total_models',
            'Total registered models'
        )
        
        self.total_benchmarks = Gauge(
            'total_benchmarks',
            'Total available benchmarks'
        )
        
        self.total_cost = Gauge(
            'total_cost_usd',
            'Total cost in USD'
        )
        
        # Resource metrics
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['type']  # type: rss, vms, available
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            ['core']  # core: total, 0, 1, ...
        )
        
        self.disk_usage = Gauge(
            'disk_usage_bytes',
            'Disk usage in bytes',
            ['type']  # type: used, free, total
        )
        
        # Error metrics
        self.errors_total = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'component', 'severity']
        )
        
        self.error_rate = Gauge(
            'error_rate',
            'Error rate (errors per second)',
            ['error_type', 'component']
        )
        
        # API metrics
        self.api_requests = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status']
        )
        
        self.api_duration = Histogram(
            'api_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        )
        
        self.api_active_requests = Gauge(
            'api_active_requests',
            'Number of active API requests',
            ['method', 'endpoint']
        )
        
        # Queue metrics
        self.queue_size = Gauge(
            'queue_size',
            'Queue size',
            ['queue_name']
        )
        
        self.queue_duration = Histogram(
            'queue_duration_seconds',
            'Time spent in queue',
            ['queue_name'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
        )
        
        # Custom metrics registry
        self.custom_metrics: Dict[str, Any] = {}
    
    def record_benchmark(
        self,
        model: str,
        benchmark: str,
        accuracy: float,
        throughput: float,
        duration: float,
        latency_p50: Optional[float] = None,
        latency_p95: Optional[float] = None,
        latency_p99: Optional[float] = None,
        total_samples: int = 0,
        correct_samples: int = 0,
        status: str = "success",
    ) -> None:
        """
        Record comprehensive benchmark metrics.
        
        Args:
            model: Model name
            benchmark: Benchmark name
            accuracy: Accuracy score
            throughput: Throughput in tokens/second
            duration: Duration in seconds
            latency_p50: P50 latency (optional)
            latency_p95: P95 latency (optional)
            latency_p99: P99 latency (optional)
            total_samples: Total samples processed
            correct_samples: Correct samples
            status: Status (success, failed, timeout)
        """
        if not self.enabled:
            return
        
        self.benchmark_requests.labels(
            model=model,
            benchmark=benchmark,
            status=status
        ).inc()
        
        self.benchmark_duration.labels(
            model=model,
            benchmark=benchmark
        ).observe(duration)
        
        self.benchmark_accuracy.labels(
            model=model,
            benchmark=benchmark
        ).set(accuracy)
        
        self.benchmark_throughput.labels(
            model=model,
            benchmark=benchmark
        ).set(throughput)
        
        # Record latency percentiles
        if latency_p50 is not None:
            self.benchmark_latency.labels(
                model=model,
                benchmark=benchmark,
                percentile="p50"
            ).observe(latency_p50)
        
        if latency_p95 is not None:
            self.benchmark_latency.labels(
                model=model,
                benchmark=benchmark,
                percentile="p95"
            ).observe(latency_p95)
        
        if latency_p99 is not None:
            self.benchmark_latency.labels(
                model=model,
                benchmark=benchmark,
                percentile="p99"
            ).observe(latency_p99)
        
        # Record samples
        if total_samples > 0:
            self.benchmark_samples.labels(
                model=model,
                benchmark=benchmark,
                result="correct"
            ).inc(correct_samples)
            
            self.benchmark_samples.labels(
                model=model,
                benchmark=benchmark,
                result="incorrect"
            ).inc(total_samples - correct_samples)
    
    def record_error(
        self,
        error_type: str,
        component: str,
        severity: str = "error",
    ) -> None:
        """
        Record error metric.
        
        Args:
            error_type: Error type (e.g., ValueError, TimeoutError)
            component: Component name (e.g., benchmark, api, model_loader)
            severity: Severity level (info, warning, error, critical)
        """
        if not self.enabled:
            return
        
        self.errors_total.labels(
            error_type=error_type,
            component=component,
            severity=severity
        ).inc()
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float,
    ) -> None:
        """
        Record API request metric.
        
        Args:
            method: HTTP method
            endpoint: Endpoint path
            status: HTTP status code
            duration: Request duration in seconds
        """
        if not self.enabled:
            return
        
        status_str = str(status)
        self.api_requests.labels(
            method=method,
            endpoint=endpoint,
            status=status_str
        ).inc()
        
        self.api_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def update_system_metrics(
        self,
        active_experiments: int,
        total_models: int,
        total_benchmarks: int,
        total_cost: float,
    ) -> None:
        """
        Update system metrics.
        
        Args:
            active_experiments: Number of active experiments
            total_models: Total registered models
            total_benchmarks: Total available benchmarks
            total_cost: Total cost in USD
        """
        if not self.enabled:
            return
        
        self.active_experiments.set(active_experiments)
        self.total_models.set(total_models)
        self.total_benchmarks.set(total_benchmarks)
        self.total_cost.set(total_cost)
    
    def update_resource_metrics(
        self,
        memory_rss: Optional[float] = None,
        memory_vms: Optional[float] = None,
        memory_available: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        cpu_per_core: Optional[List[float]] = None,
        disk_used: Optional[float] = None,
        disk_free: Optional[float] = None,
        disk_total: Optional[float] = None,
    ) -> None:
        """
        Update resource metrics.
        
        Args:
            memory_rss: RSS memory in bytes
            memory_vms: VMS memory in bytes
            memory_available: Available memory in bytes
            cpu_percent: CPU usage percentage
            cpu_per_core: CPU usage per core
            disk_used: Disk used in bytes
            disk_free: Disk free in bytes
            disk_total: Disk total in bytes
        """
        if not self.enabled:
            return
        
        if memory_rss is not None:
            self.memory_usage.labels(type="rss").set(memory_rss)
        
        if memory_vms is not None:
            self.memory_usage.labels(type="vms").set(memory_vms)
        
        if memory_available is not None:
            self.memory_usage.labels(type="available").set(memory_available)
        
        if cpu_percent is not None:
            self.cpu_usage.labels(core="total").set(cpu_percent)
        
        if cpu_per_core:
            for idx, percent in enumerate(cpu_per_core):
                self.cpu_usage.labels(core=str(idx)).set(percent)
        
        if disk_used is not None:
            self.disk_usage.labels(type="used").set(disk_used)
        
        if disk_free is not None:
            self.disk_usage.labels(type="free").set(disk_free)
        
        if disk_total is not None:
            self.disk_usage.labels(type="total").set(disk_total)
    
    def update_queue_metrics(
        self,
        queue_name: str,
        size: int,
        duration: Optional[float] = None,
    ) -> None:
        """
        Update queue metrics.
        
        Args:
            queue_name: Queue name
            size: Current queue size
            duration: Time spent in queue (optional)
        """
        if not self.enabled:
            return
        
        self.queue_size.labels(queue_name=queue_name).set(size)
        
        if duration is not None:
            self.queue_duration.labels(queue_name=queue_name).observe(duration)
    
    def create_custom_metric(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Any:
        """
        Create a custom metric.
        
        Args:
            name: Metric name
            description: Metric description
            metric_type: Metric type
            labels: Optional label names
            buckets: Optional buckets for histograms
        
        Returns:
            Prometheus metric object
        """
        if not self.enabled:
            return None
        
        labels = labels or []
        
        if metric_type == MetricType.COUNTER:
            metric = Counter(name, description, labels)
        elif metric_type == MetricType.GAUGE:
            metric = Gauge(name, description, labels)
        elif metric_type == MetricType.HISTOGRAM:
            metric = Histogram(name, description, labels, buckets=buckets)
        elif metric_type == MetricType.SUMMARY:
            metric = Summary(name, description, labels)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        self.custom_metrics[name] = metric
        return metric
    
    def export_metrics(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        if not self.enabled:
            return "# Prometheus metrics not available\n"
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return f"# Error exporting metrics: {e}\n"
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        if not self.enabled:
            return {"enabled": False}
        
        summary = {
            "enabled": True,
            "timestamp": datetime.now().isoformat(),
            "custom_metrics": list(self.custom_metrics.keys()),
            "metric_types": {
                "benchmark": ["requests", "duration", "accuracy", "throughput", "latency", "samples"],
                "system": ["experiments", "models", "benchmarks", "cost"],
                "resources": ["memory", "cpu", "disk"],
                "errors": ["total", "rate"],
                "api": ["requests", "duration", "active"],
                "queue": ["size", "duration"],
            }
        }
        
        return summary
    
    def start_metrics_server(self, port: int = 8000) -> None:
        """
        Start Prometheus metrics HTTP server.
        
        Args:
            port: Port to listen on
        """
        if not self.enabled:
            logger.warning("Cannot start metrics server: Prometheus not available")
            return
        
        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


__all__ = [
    "MetricType",
    "MetricValue",
    "MetricDefinition",
    "MetricsCollector",
    "metrics_collector",
    "PROMETHEUS_AVAILABLE",
]
