"""
Advanced Metrics System for Blaze AI

This module provides comprehensive metrics collection, Prometheus integration,
and advanced monitoring capabilities for production environments.
"""

from __future__ import annotations

import time
import asyncio
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum
import weakref

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info, Enum as PromEnum,
        generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    class PromEnum:
        def __init__(self, *args, **kwargs): pass
        def state(self, *args, **kwargs): pass

from ..core.interfaces import CoreConfig
from .logging import get_logger

# =============================================================================
# Metrics Types and Enums
# =============================================================================

class MetricType(Enum):
    """Metric types supported by the system."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"
    ENUM = "enum"

class MetricCategory(Enum):
    """Categories for organizing metrics."""
    SYSTEM = "system"
    ENGINE = "engine"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SECURITY = "security"
    CUSTOM = "custom"

@dataclass
class MetricDefinition:
    """Definition of a metric with metadata."""
    name: str
    description: str
    metric_type: MetricType
    category: MetricCategory
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    quantiles: Optional[List[float]] = None
    unit: Optional[str] = None
    help_text: Optional[str] = None

# =============================================================================
# Advanced Metrics Collector
# =============================================================================

class AdvancedMetricsCollector:
    """Advanced metrics collector with Prometheus integration."""
    
    def __init__(self, config: Optional[CoreConfig] = None):
        self.config = config
        self.logger = get_logger("metrics_collector")
        
        # Prometheus registry
        self.registry = CollectorRegistry()
        self.prometheus_available = PROMETHEUS_AVAILABLE
        
        # Metrics storage
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Performance tracking
        self.collection_times: List[float] = []
        self.last_collection = time.time()
        
        # Background collection
        self._collection_task: Optional[asyncio.Task] = None
        self._start_background_collection()
    
    def _start_background_collection(self):
        """Start background metrics collection."""
        if self._collection_task is None or self._collection_task.done():
            self._collection_task = asyncio.create_task(self._background_collection_loop())
    
    async def _background_collection_loop(self):
        """Background metrics collection loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                await self._collect_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background collection error: {e}")
                await asyncio.sleep(300)  # Wait longer on error
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            start_time = time.time()
            
            # Collect system metrics
            await self._collect_memory_metrics()
            await self._collect_cpu_metrics()
            await self._collect_network_metrics()
            await self._collect_disk_metrics()
            
            collection_time = time.time() - start_time
            self.collection_times.append(collection_time)
            
            # Keep only last 100 collection times
            if len(self.collection_times) > 100:
                self.collection_times = self.collection_times[-100:]
            
            self.last_collection = time.time()
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
    
    async def _collect_memory_metrics(self):
        """Collect memory usage metrics."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            self.set_gauge("system_memory_usage_bytes", memory.used, 
                          labels=["type:ram"], description="System RAM usage in bytes")
            self.set_gauge("system_memory_available_bytes", memory.available,
                          labels=["type:ram"], description="System available RAM in bytes")
            self.set_gauge("system_memory_percent", memory.percent,
                          labels=["type:ram"], description="System RAM usage percentage")
            self.set_gauge("system_swap_usage_bytes", swap.used,
                          labels=["type:swap"], description="System swap usage in bytes")
            
        except ImportError:
            self.logger.warning("psutil not available, skipping memory metrics")
        except Exception as e:
            self.logger.error(f"Memory metrics collection failed: {e}")
    
    async def _collect_cpu_metrics(self):
        """Collect CPU usage metrics."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            self.set_gauge("system_cpu_usage_percent", cpu_percent,
                          labels=["type:overall"], description="System CPU usage percentage")
            self.set_gauge("system_cpu_count", cpu_count,
                          labels=["type:cores"], description="Number of CPU cores")
            
            if cpu_freq:
                self.set_gauge("system_cpu_frequency_mhz", cpu_freq.current,
                              labels=["type:current"], description="Current CPU frequency in MHz")
            
        except ImportError:
            self.logger.warning("psutil not available, skipping CPU metrics")
        except Exception as e:
            self.logger.error(f"CPU metrics collection failed: {e}")
    
    async def _collect_network_metrics(self):
        """Collect network usage metrics."""
        try:
            import psutil
            
            net_io = psutil.net_io_counters()
            
            self.set_gauge("system_network_bytes_sent", net_io.bytes_sent,
                          labels=["direction:out"], description="Total bytes sent")
            self.set_gauge("system_network_bytes_recv", net_io.bytes_recv,
                          labels=["direction:in"], description="Total bytes received")
            self.set_gauge("system_network_packets_sent", net_io.packets_sent,
                          labels=["direction:out"], description="Total packets sent")
            self.set_gauge("system_network_packets_recv", net_io.packets_recv,
                          labels=["direction:in"], description="Total packets received")
            
        except ImportError:
            self.logger.warning("psutil not available, skipping network metrics")
        except Exception as e:
            self.logger.error(f"Network metrics collection failed: {e}")
    
    async def _collect_disk_metrics(self):
        """Collect disk usage metrics."""
        try:
            import psutil
            
            disk_usage = psutil.disk_usage('/')
            
            self.set_gauge("system_disk_usage_bytes", disk_usage.used,
                          labels=["mount:/"], description="Disk usage in bytes")
            self.set_gauge("system_disk_free_bytes", disk_usage.free,
                          labels=["mount:/"], description="Disk free space in bytes")
            self.set_gauge("system_disk_percent", disk_usage.percent,
                          labels=["mount:/"], description="Disk usage percentage")
            
        except ImportError:
            self.logger.warning("psutil not available, skipping disk metrics")
        except Exception as e:
            self.logger.error(f"Disk metrics collection failed: {e}")
    
    def define_metric(self, definition: MetricDefinition):
        """Define a new metric with metadata."""
        try:
            if definition.name in self.metrics:
                self.logger.warning(f"Metric {definition.name} already defined, skipping")
                return
            
            # Create Prometheus metric based on type
            if definition.metric_type == MetricType.COUNTER:
                metric = Counter(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.HISTOGRAM:
                metric = Histogram(
                    definition.name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets or [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.SUMMARY:
                metric = Summary(
                    definition.name,
                    definition.description,
                    definition.labels,
                    quantiles=definition.quantiles or [0.5, 0.9, 0.99],
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.INFO:
                metric = Info(
                    definition.name,
                    definition.description,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.ENUM:
                metric = PromEnum(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            else:
                raise ValueError(f"Unsupported metric type: {definition.metric_type}")
            
            self.metrics[definition.name] = metric
            self.metric_definitions[definition.name] = definition
            
            self.logger.info(f"Metric defined: {definition.name} ({definition.metric_type.value})")
            
        except Exception as e:
            self.logger.error(f"Failed to define metric {definition.name}: {e}")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[List[str]] = None):
        """Increment a counter metric."""
        try:
            if name in self.metrics:
                metric = self.metrics[name]
                if hasattr(metric, 'inc'):
                    metric.inc(value)
                else:
                    self.logger.warning(f"Metric {name} is not a counter")
            else:
                self.logger.warning(f"Metric {name} not found")
        except Exception as e:
            self.logger.error(f"Failed to increment counter {name}: {e}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[List[str]] = None, 
                  description: Optional[str] = None):
        """Set a gauge metric value."""
        try:
            if name in self.metrics:
                metric = self.metrics[name]
                if hasattr(metric, 'set'):
                    metric.set(value)
                else:
                    self.logger.warning(f"Metric {name} is not a gauge")
            else:
                # Auto-create gauge if not defined
                if description:
                    definition = MetricDefinition(
                        name=name,
                        description=description,
                        metric_type=MetricType.GAUGE,
                        category=MetricCategory.SYSTEM,
                        labels=labels or []
                    )
                    self.define_metric(definition)
                    self.metrics[name].set(value)
                else:
                    self.logger.warning(f"Metric {name} not found and no description provided")
        except Exception as e:
            self.logger.error(f"Failed to set gauge {name}: {e}")
    
    def observe_histogram(self, name: str, value: float, labels: Optional[List[str]] = None):
        """Observe a value in a histogram metric."""
        try:
            if name in self.metrics:
                metric = self.metrics[name]
                if hasattr(metric, 'observe'):
                    metric.observe(value)
                else:
                    self.logger.warning(f"Metric {name} is not a histogram")
            else:
                self.logger.warning(f"Metric {name} not found")
        except Exception as e:
            self.logger.error(f"Failed to observe histogram {name}: {e}")
    
    def observe_summary(self, name: str, value: float, labels: Optional[List[str]] = None):
        """Observe a value in a summary metric."""
        try:
            if name in self.metrics:
                metric = self.metrics[name]
                if hasattr(metric, 'observe'):
                    metric.observe(value)
                else:
                    self.logger.warning(f"Metric {name} is not a summary")
            else:
                self.logger.warning(f"Metric {name} not found")
        except Exception as e:
            self.logger.error(f"Failed to observe summary {name}: {e}")
    
    def set_info(self, name: str, info_dict: Dict[str, str]):
        """Set info metric values."""
        try:
            if name in self.metrics:
                metric = self.metrics[name]
                if hasattr(metric, 'info'):
                    metric.info(info_dict)
                else:
                    self.logger.warning(f"Metric {name} is not an info metric")
            else:
                self.logger.warning(f"Metric {name} not found")
        except Exception as e:
            self.logger.error(f"Failed to set info {name}: {e}")
    
    def set_enum_state(self, name: str, state: str):
        """Set enum metric state."""
        try:
            if name in self.metrics:
                metric = self.metrics[name]
                if hasattr(metric, 'state'):
                    metric.state(state)
                else:
                    self.logger.warning(f"Metric {name} is not an enum metric")
            else:
                self.logger.warning(f"Metric {name} not found")
        except Exception as e:
            self.logger.error(f"Failed to set enum state {name}: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {
            "total_metrics": len(self.metrics),
            "metric_types": {},
            "categories": {},
            "collection_stats": {
                "last_collection": self.last_collection,
                "average_collection_time": sum(self.collection_times) / len(self.collection_times) if self.collection_times else 0,
                "total_collections": len(self.collection_times)
            },
            "prometheus_available": self.prometheus_available
        }
        
        # Count by type
        for definition in self.metric_definitions.values():
            metric_type = definition.metric_type.value
            summary["metric_types"][metric_type] = summary["metric_types"].get(metric_type, 0) + 1
            
            category = definition.category.value
            summary["categories"][category] = summary["categories"].get(category, 0) + 1
        
        return summary
    
    def generate_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics output."""
        if not self.prometheus_available:
            return "# Prometheus client not available\n"
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to generate Prometheus metrics: {e}")
            return f"# Error generating metrics: {e}\n"
    
    async def shutdown(self):
        """Shutdown the metrics collector."""
        self.logger.info("Shutting down metrics collector...")
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Metrics collector shutdown complete")

# =============================================================================
# Performance Metrics Context Manager
# =============================================================================

@asynccontextmanager
async def track_performance(metrics_collector: AdvancedMetricsCollector, 
                           operation_name: str, 
                           labels: Optional[List[str]] = None):
    """Context manager for tracking operation performance."""
    start_time = time.time()
    start_cpu = time.process_time()
    
    try:
        yield
        success = True
    except Exception as e:
        success = False
        raise
    finally:
        duration = time.time() - start_time
        cpu_time = time.process_time() - start_cpu
        
        # Record performance metrics
        metrics_collector.observe_histogram(
            f"{operation_name}_duration_seconds", 
            duration, 
            labels
        )
        metrics_collector.observe_histogram(
            f"{operation_name}_cpu_time_seconds", 
            cpu_time, 
            labels
        )
        metrics_collector.increment_counter(
            f"{operation_name}_total", 
            1.0, 
            labels
        )
        metrics_collector.increment_counter(
            f"{operation_name}_{'success' if success else 'failure'}", 
            1.0, 
            labels
        )

# =============================================================================
# Global Metrics Instance
# =============================================================================

_global_metrics_collector: Optional[AdvancedMetricsCollector] = None

def get_metrics_collector(config: Optional[CoreConfig] = None) -> AdvancedMetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = AdvancedMetricsCollector(config)
    return _global_metrics_collector

async def shutdown_metrics_collector():
    """Shutdown the global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector:
        await _global_metrics_collector.shutdown()
        _global_metrics_collector = None


