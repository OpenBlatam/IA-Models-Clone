"""
Cache telemetry and observability.

Provides comprehensive telemetry for production monitoring.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from collections import deque

from kv_cache.types import StatsDict

logger = logging.getLogger(__name__)


class CacheTelemetry:
    """
    Cache telemetry collector.
    
    Collects comprehensive telemetry data for observability.
    """
    
    def __init__(
        self,
        cache: Any,
        export_interval: int = 60,
        exporters: Optional[List[Callable]] = None
    ):
        """
        Initialize cache telemetry.
        
        Args:
            cache: Cache instance
            export_interval: Interval for exporting metrics (seconds)
            exporters: List of exporter functions
        """
        self.cache = cache
        self.export_interval = export_interval
        self.exporters = exporters or []
        
        self.metrics_buffer: deque = deque(maxlen=1000)
        self.last_export = time.time()
        self.total_exports = 0
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        metric = {
            "timestamp": time.time(),
            "name": name,
            "value": value,
            "tags": tags or {}
        }
        
        self.metrics_buffer.append(metric)
        
        # Auto-export if interval reached
        if time.time() - self.last_export >= self.export_interval:
            self.export_metrics()
    
    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True
    ) -> None:
        """
        Record operation metrics.
        
        Args:
            operation: Operation name
            duration: Operation duration (seconds)
            success: Whether operation succeeded
        """
        self.record_metric(
            f"cache.operation.{operation}.duration",
            duration * 1000,  # Convert to ms
            tags={"success": str(success)}
        )
        
        self.record_metric(
            f"cache.operation.{operation}.count",
            1.0,
            tags={"success": str(success)}
        )
    
    def record_cache_stats(self) -> None:
        """Record current cache statistics."""
        stats = self.cache.get_stats()
        
        self.record_metric("cache.hit_rate", stats.get("hit_rate", 0.0))
        self.record_metric("cache.num_entries", float(stats.get("num_entries", 0)))
        self.record_metric("cache.memory_mb", stats.get("storage_memory_mb", 0.0))
        self.record_metric("cache.evictions", float(stats.get("evictions", 0)))
        self.record_metric("cache.hits", float(stats.get("hits", 0)))
        self.record_metric("cache.misses", float(stats.get("misses", 0)))
    
    def export_metrics(self) -> int:
        """
        Export metrics to all exporters.
        
        Returns:
            Number of metrics exported
        """
        if not self.metrics_buffer:
            return 0
        
        metrics_to_export = list(self.metrics_buffer)
        self.metrics_buffer.clear()
        
        exported = 0
        for exporter in self.exporters:
            try:
                exporter(metrics_to_export)
                exported += len(metrics_to_export)
            except Exception as e:
                logger.warning(f"Exporter failed: {e}")
        
        self.last_export = time.time()
        self.total_exports += 1
        
        return exported
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Dictionary with metrics summary
        """
        return {
            "buffered_metrics": len(self.metrics_buffer),
            "total_exports": self.total_exports,
            "last_export": self.last_export,
            "exporters": len(self.exporters)
        }


class PrometheusExporter:
    """Prometheus metrics exporter."""
    
    def __init__(self, registry: Any = None):
        """
        Initialize Prometheus exporter.
        
        Args:
            registry: Prometheus registry (optional)
        """
        self.registry = registry
        try:
            from prometheus_client import Counter, Histogram, Gauge
            self.Counter = Counter
            self.Histogram = Histogram
            self.Gauge = Gauge
            self._init_metrics()
        except ImportError:
            logger.warning("prometheus_client not available")
            self.Counter = None
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        if self.Counter is None:
            return
        
        self.cache_hits = self.Counter('cache_hits_total', 'Total cache hits')
        self.cache_misses = self.Counter('cache_misses_total', 'Total cache misses')
        self.cache_operations = self.Histogram(
            'cache_operation_duration_seconds',
            'Cache operation duration',
            ['operation']
        )
        self.cache_size = self.Gauge('cache_size', 'Current cache size')
        self.cache_memory = self.Gauge('cache_memory_mb', 'Cache memory usage in MB')
    
    def export(self, metrics: List[Dict[str, Any]]) -> None:
        """
        Export metrics to Prometheus.
        
        Args:
            metrics: List of metric dictionaries
        """
        if self.Counter is None:
            return
        
        for metric in metrics:
            name = metric["name"]
            value = metric["value"]
            
            if "cache.hits" in name:
                self.cache_hits.inc(value)
            elif "cache.misses" in name:
                self.cache_misses.inc(value)
            elif "cache.operation" in name and "duration" in name:
                operation = name.split(".")[-2]
                self.cache_operations.labels(operation=operation).observe(value / 1000)
            elif "cache.num_entries" in name:
                self.cache_size.set(value)
            elif "cache.memory_mb" in name:
                self.cache_memory.set(value)


class StatsDExporter:
    """StatsD metrics exporter."""
    
    def __init__(self, client: Any = None):
        """
        Initialize StatsD exporter.
        
        Args:
            client: StatsD client (optional)
        """
        self.client = client
        if client is None:
            try:
                from statsd import StatsClient
                self.client = StatsClient()
            except ImportError:
                logger.warning("statsd not available")
    
    def export(self, metrics: List[Dict[str, Any]]) -> None:
        """
        Export metrics to StatsD.
        
        Args:
            metrics: List of metric dictionaries
        """
        if self.client is None:
            return
        
        for metric in metrics:
            name = metric["name"].replace(".", "_")
            value = metric["value"]
            
            if "duration" in name or "latency" in name:
                self.client.timing(name, value)
            else:
                self.client.gauge(name, value)

