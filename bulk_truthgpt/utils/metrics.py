"""
Advanced Metrics System
======================

Comprehensive metrics collection and monitoring for the Bulk TruthGPT system.
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import numpy as np

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricValue:
    """Metric value with metadata."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricDefinition:
    """Metric definition."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    quantiles: Optional[List[float]] = None

class MetricsCollector:
    """
    Advanced metrics collector.
    
    Collects system metrics, application metrics, and custom metrics
    with support for Prometheus integration.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.metrics = {}
        self.custom_metrics = {}
        self.collection_interval = 60  # seconds
        self.retention_days = 7
        self._running = False
        self._collection_task = None
        self._lock = threading.Lock()
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Initialize system metrics
        self._init_system_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Request metrics
        self.request_count = Counter(
            'bulk_truthgpt_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'bulk_truthgpt_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )
        
        # Generation metrics
        self.generation_count = Counter(
            'bulk_truthgpt_generations_total',
            'Total number of document generations',
            ['task_id', 'status']
        )
        
        self.generation_duration = Histogram(
            'bulk_truthgpt_generation_duration_seconds',
            'Generation duration in seconds',
            ['task_id'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
        )
        
        self.generation_quality = Histogram(
            'bulk_truthgpt_generation_quality_score',
            'Generation quality score',
            ['task_id'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'bulk_truthgpt_system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.system_memory_usage = Gauge(
            'bulk_truthgpt_system_memory_usage_percent',
            'System memory usage percentage'
        )
        
        self.system_disk_usage = Gauge(
            'bulk_truthgpt_system_disk_usage_percent',
            'System disk usage percentage'
        )
        
        # Application metrics
        self.active_tasks = Gauge(
            'bulk_truthgpt_active_tasks',
            'Number of active tasks'
        )
        
        self.queue_size = Gauge(
            'bulk_truthgpt_queue_size',
            'Size of the task queue'
        )
        
        self.cache_hit_rate = Gauge(
            'bulk_truthgpt_cache_hit_rate',
            'Cache hit rate'
        )
        
        # Error metrics
        self.error_count = Counter(
            'bulk_truthgpt_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        # Performance metrics
        self.throughput = Gauge(
            'bulk_truthgpt_throughput_documents_per_second',
            'Document generation throughput'
        )
        
        self.latency_p50 = Gauge(
            'bulk_truthgpt_latency_p50_seconds',
            '50th percentile latency'
        )
        
        self.latency_p95 = Gauge(
            'bulk_truthgpt_latency_p95_seconds',
            '95th percentile latency'
        )
        
        self.latency_p99 = Gauge(
            'bulk_truthgpt_latency_p99_seconds',
            '99th percentile latency'
        )
    
    def _init_system_metrics(self):
        """Initialize system metrics collection."""
        self.system_metrics = {
            'cpu_percent': deque(maxlen=1000),
            'memory_percent': deque(maxlen=1000),
            'disk_percent': deque(maxlen=1000),
            'network_bytes_sent': deque(maxlen=1000),
            'network_bytes_recv': deque(maxlen=1000),
            'process_count': deque(maxlen=1000),
            'load_average': deque(maxlen=1000)
        }
    
    async def start(self):
        """Start metrics collection."""
        if self._running:
            logger.warning("Metrics collector is already running")
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collect_metrics())
        logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop metrics collection."""
        if not self._running:
            logger.warning("Metrics collector is not running")
            return
        
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics collector stopped")
    
    async def _collect_metrics(self):
        """Collect metrics in background."""
        while self._running:
            try:
                await asyncio.sleep(self.collection_interval)
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await self._cleanup_old_metrics()
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            self.system_metrics['cpu_percent'].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.percent)
            self.system_metrics['memory_percent'].append(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_disk_usage.set(disk.percent)
            self.system_metrics['disk_percent'].append(disk.percent)
            
            # Network usage
            network = psutil.net_io_counters()
            self.system_metrics['network_bytes_sent'].append(network.bytes_sent)
            self.system_metrics['network_bytes_recv'].append(network.bytes_recv)
            
            # Process count
            process_count = len(psutil.pids())
            self.system_metrics['process_count'].append(process_count)
            
            # Load average
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            self.system_metrics['load_average'].append(load_avg)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    async def _collect_application_metrics(self):
        """Collect application metrics."""
        try:
            # This would be implemented based on specific application metrics
            # For now, we'll just log that we're collecting
            pass
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {str(e)}")
    
    async def _cleanup_old_metrics(self):
        """Cleanup old metrics data."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
            
            # Cleanup custom metrics
            for metric_name, values in self.custom_metrics.items():
                self.custom_metrics[metric_name] = [
                    v for v in values if v.timestamp > cutoff_time
                ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {str(e)}")
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record request metrics."""
        self.request_count.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_generation(self, task_id: str, status: str, duration: float, quality: float):
        """Record generation metrics."""
        self.generation_count.labels(task_id=task_id, status=status).inc()
        self.generation_duration.labels(task_id=task_id).observe(duration)
        self.generation_quality.labels(task_id=task_id).observe(quality)
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        self.error_count.labels(error_type=error_type, component=component).inc()
    
    def update_active_tasks(self, count: int):
        """Update active tasks count."""
        self.active_tasks.set(count)
    
    def update_queue_size(self, size: int):
        """Update queue size."""
        self.queue_size.set(size)
    
    def update_cache_hit_rate(self, rate: float):
        """Update cache hit rate."""
        self.cache_hit_rate.set(rate)
    
    def update_throughput(self, throughput: float):
        """Update throughput."""
        self.throughput.set(throughput)
    
    def update_latency(self, p50: float, p95: float, p99: float):
        """Update latency percentiles."""
        self.latency_p50.set(p50)
        self.latency_p95.set(p95)
        self.latency_p99.set(p99)
    
    def record_custom_metric(self, name: str, value: Union[int, float], 
                           labels: Optional[Dict[str, str]] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Record custom metric."""
        metric_value = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            if name not in self.custom_metrics:
                self.custom_metrics[name] = deque(maxlen=10000)
            self.custom_metrics[name].append(metric_value)
    
    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metric summary."""
        with self._lock:
            if name not in self.custom_metrics:
                return None
            
            values = [v.value for v in self.custom_metrics[name]]
            if not values:
                return None
            
            return {
                'name': name,
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            return {
                'system_metrics': {
                    name: list(values) for name, values in self.system_metrics.items()
                },
                'custom_metrics': {
                    name: [v.__dict__ for v in values] for name, values in self.custom_metrics.items()
                }
            }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format == "json":
            import json
            return json.dumps(self.get_all_metrics(), indent=2, default=str)
        elif format == "prometheus":
            return self.get_prometheus_metrics()
        else:
            return str(self.get_all_metrics())

class MetricsContext:
    """
    Context manager for metrics collection.
    
    Automatically records timing and other metrics for operations.
    """
    
    def __init__(self, collector: MetricsCollector, operation: str, **labels):
        self.collector = collector
        self.operation = operation
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_custom_metric(
                f"{self.operation}_duration",
                duration,
                self.labels
            )
            
            if exc_type:
                self.collector.record_custom_metric(
                    f"{self.operation}_errors",
                    1,
                    {**self.labels, 'error_type': exc_type.__name__}
                )
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_custom_metric(
                f"{self.operation}_duration",
                duration,
                self.labels
            )
            
            if exc_type:
                self.collector.record_custom_metric(
                    f"{self.operation}_errors",
                    1,
                    {**self.labels, 'error_type': exc_type.__name__}
                )

def metrics_context(collector: MetricsCollector, operation: str, **labels):
    """Create a metrics context."""
    return MetricsContext(collector, operation, **labels)

# Global metrics collector
metrics_collector = MetricsCollector()











