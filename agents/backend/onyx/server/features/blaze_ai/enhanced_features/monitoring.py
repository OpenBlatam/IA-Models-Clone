"""
Enhanced Performance Monitoring Module for Blaze AI.

This module provides comprehensive performance monitoring, metrics collection,
and system health monitoring capabilities.
"""

import asyncio
import gc
import inspect
import json
import os
import psutil
import time
import tracemalloc
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from weakref import WeakKeyDictionary

from pydantic import BaseModel, Field

from core.config import MonitoringConfig
from core.exceptions import ServiceUnavailableError
from core.logging import get_logger


# ============================================================================
# MONITORING MODELS AND CONFIGURATION
# ============================================================================

class MetricValue(BaseModel):
    """Represents a single metric value with timestamp."""
    value: Union[int, float] = Field(..., description="Metric value")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the metric")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Metric(BaseModel):
    """Represents a metric with multiple values over time."""
    name: str = Field(..., description="Metric name")
    description: str = Field(..., description="Metric description")
    unit: str = Field(default="", description="Metric unit")
    type: str = Field(..., description="Metric type (counter, gauge, histogram)")
    values: List[MetricValue] = Field(default_factory=list, description="Metric values")
    max_history: int = Field(default=1000, description="Maximum number of values to keep")
    
    def add_value(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add a new value to the metric."""
        metric_value = MetricValue(
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        self.values.append(metric_value)
        
        # Maintain history size
        if len(self.values) > self.max_history:
            self.values.pop(0)
    
    def get_latest_value(self) -> Optional[MetricValue]:
        """Get the most recent metric value."""
        return self.values[-1] if self.values else None
    
    def get_average(self, window_minutes: int = 5) -> Optional[float]:
        """Get average value over a time window."""
        if not self.values:
            return None
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_values = [v.value for v in self.values if v.timestamp > cutoff_time]
        
        if not recent_values:
            return None
        
        return sum(recent_values) / len(recent_values)


class SystemMetrics(BaseModel):
    """System-level metrics."""
    cpu_percent: float = Field(0.0, description="CPU usage percentage")
    memory_percent: float = Field(0.0, description="Memory usage percentage")
    memory_available: int = Field(0, description="Available memory in bytes")
    memory_used: int = Field(0, description="Used memory in bytes")
    disk_percent: float = Field(0.0, description="Disk usage percentage")
    disk_free: int = Field(0, description="Free disk space in bytes")
    network_bytes_sent: int = Field(0, description="Network bytes sent")
    network_bytes_recv: int = Field(0, description="Network bytes received")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the metrics")


class PerformanceProfile(BaseModel):
    """Performance profile for a function or operation."""
    function_name: str = Field(..., description="Name of the function")
    module_name: str = Field(..., description="Module containing the function")
    execution_count: int = Field(0, description="Number of times executed")
    total_execution_time: float = Field(0.0, description="Total execution time in seconds")
    average_execution_time: float = Field(0.0, description="Average execution time in seconds")
    min_execution_time: float = Field(0.0, description="Minimum execution time in seconds")
    max_execution_time: float = Field(0.0, description="Maximum execution time in seconds")
    memory_usage_start: int = Field(0, description="Memory usage at start in bytes")
    memory_usage_end: int = Field(0, description="Memory usage at end in bytes")
    memory_peak: int = Field(0, description="Peak memory usage in bytes")
    last_execution: Optional[datetime] = Field(None, description="Last execution timestamp")
    errors: int = Field(0, description="Number of errors encountered")
    success_rate: float = Field(1.0, description="Success rate (0.0 to 1.0)")


class AlertRule(BaseModel):
    """Rule for generating performance alerts."""
    name: str = Field(..., description="Alert rule name")
    metric_name: str = Field(..., description="Metric to monitor")
    condition: str = Field(..., description="Alert condition (e.g., '>', '<', '==')")
    threshold: float = Field(..., description="Threshold value")
    severity: str = Field(default="warning", description="Alert severity")
    message: str = Field(..., description="Alert message")
    enabled: bool = Field(default=True, description="Whether the alert is enabled")
    cooldown_minutes: int = Field(default=5, description="Cooldown period between alerts")


class Alert(BaseModel):
    """Represents a performance alert."""
    alert_id: str = Field(default_factory=lambda: str(time.time()), description="Unique alert identifier")
    rule_name: str = Field(..., description="Name of the alert rule")
    metric_name: str = Field(..., description="Metric that triggered the alert")
    current_value: float = Field(..., description="Current metric value")
    threshold: float = Field(..., description="Threshold that was exceeded")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Alert timestamp")
    acknowledged: bool = Field(default=False, description="Whether the alert has been acknowledged")


# ============================================================================
# MONITORING INTERFACES AND BASE CLASSES
# ============================================================================

class MetricsCollector(ABC):
    """Abstract base class for metrics collection."""
    
    @abstractmethod
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics."""
        pass
    
    @abstractmethod
    async def get_metric(self, name: str) -> Optional[Metric]:
        """Get a specific metric by name."""
        pass
    
    @abstractmethod
    async def add_metric(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None) -> None:
        """Add a new metric value."""
        pass


class SystemMonitor(ABC):
    """Abstract base class for system monitoring."""
    
    @abstractmethod
    async def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        pass
    
    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start system monitoring."""
        pass
    
    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        pass


class PerformanceProfiler(ABC):
    """Abstract base class for performance profiling."""
    
    @abstractmethod
    async def profile_function(self, func: Callable) -> Callable:
        """Profile a function and return wrapped version."""
        pass
    
    @abstractmethod
    async def get_profile(self, function_name: str) -> Optional[PerformanceProfile]:
        """Get performance profile for a function."""
        pass
    
    @abstractmethod
    async def clear_profiles(self) -> None:
        """Clear all performance profiles."""
        pass


class AlertManager(ABC):
    """Abstract base class for alert management."""
    
    @abstractmethod
    async def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        pass
    
    @abstractmethod
    async def check_alerts(self) -> List[Alert]:
        """Check for new alerts based on current metrics."""
        pass
    
    @abstractmethod
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        pass


# ============================================================================
# IMPLEMENTATIONS
# ============================================================================

class InMemoryMetricsCollector(MetricsCollector):
    """In-memory metrics collector implementation."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.metrics: Dict[str, Metric] = {}
        self._lock = asyncio.Lock()
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect all current metrics."""
        async with self._lock:
            return {
                name: {
                    "name": metric.name,
                    "description": metric.description,
                    "unit": metric.unit,
                    "type": metric.type,
                    "latest_value": metric.get_latest_value().dict() if metric.get_latest_value() else None,
                    "values_count": len(metric.values),
                    "average_5min": metric.get_average(5),
                    "average_1hour": metric.get_average(60)
                }
                for name, metric in self.metrics.items()
            }
    
    async def get_metric(self, name: str) -> Optional[Metric]:
        """Get a specific metric by name."""
        async with self._lock:
            return self.metrics.get(name)
    
    async def add_metric(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None) -> None:
        """Add a new metric value."""
        async with self._lock:
            if name not in self.metrics:
                self.logger.warning(f"Metric '{name}' not found, creating default")
                self.metrics[name] = Metric(
                    name=name,
                    description=f"Auto-created metric: {name}",
                    type="gauge",
                    max_history=self.config.max_metrics_history
                )
            
            self.metrics[name].add_value(value, labels)
    
    async def create_counter(self, name: str, description: str, unit: str = "") -> None:
        """Create a new counter metric."""
        async with self._lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(
                    name=name,
                    description=description,
                    unit=unit,
                    type="counter",
                    max_history=self.config.max_metrics_history
                )
                self.logger.info(f"Created counter metric: {name}")
    
    async def create_gauge(self, name: str, description: str, unit: str = "") -> None:
        """Create a new gauge metric."""
        async with self._lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(
                    name=name,
                    description=description,
                    unit=unit,
                    type="gauge",
                    max_history=self.config.max_metrics_history
                )
                self.logger.info(f"Created gauge metric: {name}")
    
    async def create_histogram(self, name: str, description: str, unit: str = "") -> None:
        """Create a new histogram metric."""
        async with self._lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(
                    name=name,
                    description=description,
                    unit=unit,
                    type="histogram",
                    max_history=self.config.max_metrics_history
                )
                self.logger.info(f"Created histogram metric: {name}")
    
    async def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        async with self._lock:
            if name in self.metrics and self.metrics[name].type == "counter":
                current_value = self.metrics[name].get_latest_value()
                new_value = (current_value.value if current_value else 0) + value
                self.metrics[name].add_value(new_value, labels)
            else:
                self.logger.warning(f"Cannot increment non-counter metric: {name}")
    
    async def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        async with self._lock:
            if name in self.metrics and self.metrics[name].type == "gauge":
                self.metrics[name].add_value(value, labels)
            else:
                self.logger.warning(f"Cannot set non-gauge metric: {name}")
    
    async def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a value in a histogram metric."""
        async with self._lock:
            if name in self.metrics and self.metrics[name].type == "histogram":
                self.metrics[name].add_value(value, labels)
            else:
                self.logger.warning(f"Cannot record in non-histogram metric: {name}")
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default system metrics."""
        default_metrics = [
            ("requests_total", "Total number of requests", "counter", ""),
            ("requests_duration_seconds", "Request duration in seconds", "histogram", "seconds"),
            ("requests_per_second", "Requests per second", "gauge", "requests/sec"),
            ("memory_usage_bytes", "Memory usage in bytes", "gauge", "bytes"),
            ("cpu_usage_percent", "CPU usage percentage", "gauge", "percent"),
            ("active_connections", "Number of active connections", "gauge", "connections"),
            ("errors_total", "Total number of errors", "counter", ""),
            ("response_time_p95", "95th percentile response time", "gauge", "seconds"),
            ("response_time_p99", "99th percentile response time", "gauge", "seconds")
        ]
        
        for name, description, metric_type, unit in default_metrics:
            if metric_type == "counter":
                self.metrics[name] = Metric(
                    name=name,
                    description=description,
                    unit=unit,
                    type=metric_type,
                    max_history=self.config.max_metrics_history
                )
            elif metric_type == "gauge":
                self.metrics[name] = Metric(
                    name=name,
                    description=description,
                    unit=unit,
                    type=metric_type,
                    max_history=self.config.max_metrics_history
                )
            elif metric_type == "histogram":
                self.metrics[name] = Metric(
                    name=name,
                    description=description,
                    unit=unit,
                    type=metric_type,
                    max_history=self.config.max_metrics_history
                )


class SystemMetricsMonitor(SystemMonitor):
    """System metrics monitoring implementation."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # System metrics cache
        self._last_metrics: Optional[SystemMetrics] = None
        self._metrics_history: deque = deque(maxlen=100)
    
    async def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_used = memory.used
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                memory_used=memory_used,
                disk_percent=disk_percent,
                disk_free=disk_free,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv
            )
            
            # Cache metrics
            self._last_metrics = metrics
            self._metrics_history.append(metrics)
            
            # Update metrics collector if available
            if self.metrics_collector:
                await self.metrics_collector.set_gauge("cpu_usage_percent", cpu_percent)
                await self.metrics_collector.set_gauge("memory_usage_bytes", memory_used)
                await self.metrics_collector.set_gauge("memory_usage_percent", memory_percent)
                await self.metrics_collector.set_gauge("disk_usage_percent", disk_percent)
                await self.metrics_collector.set_gauge("disk_free_bytes", disk_free)
                await self.metrics_collector.set_gauge("network_bytes_sent", network_bytes_sent)
                await self.metrics_collector.set_gauge("network_bytes_recv", network_bytes_recv)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            # Return cached metrics if available
            return self._last_metrics or SystemMetrics()
    
    async def start_monitoring(self) -> None:
        """Start system monitoring."""
        if self.is_monitoring:
            self.logger.warning("System monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("System monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self.get_system_metrics()
                await asyncio.sleep(self.config.metrics_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    def set_metrics_collector(self, collector: MetricsCollector) -> None:
        """Set the metrics collector for updating metrics."""
        self.metrics_collector = collector
    
    async def get_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get system metrics history."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [m for m in self._metrics_history if m.timestamp > cutoff_time]


class PerformanceProfilerImpl(PerformanceProfiler):
    """Performance profiling implementation."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.profiles: Dict[str, PerformanceProfile] = {}
        self._lock = asyncio.Lock()
        
        # Enable memory tracking if configured
        if self.config.enable_profiling:
            tracemalloc.start()
    
    async def profile_function(self, func: Callable) -> Callable:
        """Profile a function and return wrapped version."""
        if not self.config.enable_profiling:
            return func
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._profile_execution(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._profile_execution_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def _profile_execution(self, func: Callable, *args, **kwargs):
        """Profile async function execution."""
        start_time = time.time()
        start_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            await self._update_profile(
                func, execution_time, memory_usage, start_memory, end_memory, success
            )
        
        return result
    
    def _profile_execution_sync(self, func: Callable, *args, **kwargs):
        """Profile sync function execution."""
        start_time = time.time()
        start_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Use asyncio.create_task for async operation
            asyncio.create_task(self._update_profile(
                func, execution_time, memory_usage, start_memory, end_memory, success
            ))
        
        return result
    
    async def _update_profile(self, func: Callable, execution_time: float, memory_usage: int,
                            start_memory: int, end_memory: int, success: bool) -> None:
        """Update performance profile for a function."""
        async with self._lock:
            function_name = func.__name__
            module_name = func.__module__
            
            if function_name not in self.profiles:
                self.profiles[function_name] = PerformanceProfile(
                    function_name=function_name,
                    module_name=module_name
                )
            
            profile = self.profiles[function_name]
            
            # Update execution statistics
            profile.execution_count += 1
            profile.total_execution_time += execution_time
            profile.average_execution_time = profile.total_execution_time / profile.execution_count
            
            if execution_time < profile.min_execution_time or profile.min_execution_time == 0:
                profile.min_execution_time = execution_time
            
            if execution_time > profile.max_execution_time:
                profile.max_execution_time = execution_time
            
            # Update memory statistics
            profile.memory_usage_start = start_memory
            profile.memory_usage_end = end_memory
            
            if memory_usage > profile.memory_peak:
                profile.memory_peak = memory_usage
            
            # Update success rate
            if not success:
                profile.errors += 1
            
            profile.success_rate = (profile.execution_count - profile.errors) / profile.execution_count
            profile.last_execution = datetime.utcnow()
    
    async def get_profile(self, function_name: str) -> Optional[PerformanceProfile]:
        """Get performance profile for a function."""
        async with self._lock:
            return self.profiles.get(function_name)
    
    async def get_all_profiles(self) -> List[PerformanceProfile]:
        """Get all performance profiles."""
        async with self._lock:
            return list(self.profiles.values())
    
    async def clear_profiles(self) -> None:
        """Clear all performance profiles."""
        async with self._lock:
            self.profiles.clear()
            self.logger.info("All performance profiles cleared")
    
    async def get_top_functions(self, limit: int = 10, sort_by: str = "execution_time") -> List[PerformanceProfile]:
        """Get top functions by specified criteria."""
        async with self._lock:
            profiles = list(self.profiles.values())
            
            if sort_by == "execution_time":
                profiles.sort(key=lambda p: p.total_execution_time, reverse=True)
            elif sort_by == "execution_count":
                profiles.sort(key=lambda p: p.execution_count, reverse=True)
            elif sort_by == "memory_peak":
                profiles.sort(key=lambda p: p.memory_peak, reverse=True)
            elif sort_by == "average_time":
                profiles.sort(key=lambda p: p.average_execution_time, reverse=True)
            
            return profiles[:limit]


class AlertManagerImpl(AlertManager):
    """Alert management implementation."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self._lock = asyncio.Lock()
        
        # Initialize default alert rules
        self._initialize_default_rules()
    
    async def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        async with self._lock:
            self.rules[rule.name] = rule
            self.logger.info(f"Added alert rule: {rule.name}")
    
    async def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        async with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                self.logger.info(f"Removed alert rule: {rule_name}")
                return True
            return False
    
    async def check_alerts(self) -> List[Alert]:
        """Check for new alerts based on current metrics."""
        # This would typically check against current metrics
        # For now, we'll return existing unacknowledged alerts
        async with self._lock:
            return [alert for alert in self.alerts if not alert.acknowledged]
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        async with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    self.logger.info(f"Acknowledged alert: {alert_id}")
                    return True
            return False
    
    async def create_alert(self, rule: AlertRule, current_value: float) -> Alert:
        """Create a new alert."""
        alert = Alert(
            rule_name=rule.name,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            message=rule.message
        )
        
        async with self._lock:
            self.alerts.append(alert)
            self.logger.warning(f"Alert created: {alert.message} (Value: {current_value}, Threshold: {rule.threshold})")
        
        return alert
    
    async def get_alerts(self, severity: Optional[str] = None, acknowledged: Optional[bool] = None) -> List[Alert]:
        """Get alerts with optional filtering."""
        async with self._lock:
            filtered_alerts = self.alerts
            
            if severity is not None:
                filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
            
            if acknowledged is not None:
                filtered_alerts = [a for a in filtered_alerts if a.acknowledged == acknowledged]
            
            return filtered_alerts
    
    async def clear_old_alerts(self, days: int = 30) -> int:
        """Clear old alerts."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with self._lock:
            initial_count = len(self.alerts)
            self.alerts = [a for a in self.alerts if a.timestamp > cutoff_date]
            cleared_count = initial_count - len(self.alerts)
            
            if cleared_count > 0:
                self.logger.info(f"Cleared {cleared_count} old alerts")
            
            return cleared_count
    
    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                condition=">",
                threshold=80.0,
                severity="warning",
                message="CPU usage is high",
                cooldown_minutes=5
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_usage_percent",
                condition=">",
                threshold=85.0,
                severity="warning",
                message="Memory usage is high",
                cooldown_minutes=5
            ),
            AlertRule(
                name="high_disk_usage",
                metric_name="disk_usage_percent",
                condition=">",
                threshold=90.0,
                severity="critical",
                message="Disk usage is critical",
                cooldown_minutes=10
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="errors_total",
                condition=">",
                threshold=100.0,
                severity="critical",
                message="Error rate is high",
                cooldown_minutes=1
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.name] = rule


# ============================================================================
# MAIN PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Main performance monitoring orchestrator."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize monitoring components
        self.metrics_collector = InMemoryMetricsCollector(config)
        self.system_monitor = SystemMetricsMonitor(config)
        self.performance_profiler = PerformanceProfilerImpl(config)
        self.alert_manager = AlertManagerImpl(config)
        
        # Set up cross-references
        self.system_monitor.set_metrics_collector(self.metrics_collector)
        
        # Monitoring state
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info("Performance monitor initialized")
    
    async def start(self) -> None:
        """Start performance monitoring."""
        if self.is_running:
            self.logger.warning("Performance monitoring already started")
            return
        
        try:
            # Start system monitoring
            await self.system_monitor.start_monitoring()
            
            # Start main monitoring loop
            self.is_running = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("Performance monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start performance monitoring: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop performance monitoring."""
        if not self.is_running:
            return
        
        try:
            # Stop system monitoring
            await self.system_monitor.stop_monitoring()
            
            # Stop main monitoring loop
            self.is_running = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Performance monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping performance monitoring: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect system metrics
                system_metrics = await self.system_monitor.get_system_metrics()
                
                # Check for alerts
                await self._check_alerts(system_metrics)
                
                # Update custom metrics
                await self._update_custom_metrics()
                
                # Wait for next iteration
                await asyncio.sleep(self.config.metrics_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def _check_alerts(self, system_metrics: SystemMetrics) -> None:
        """Check for system alerts."""
        try:
            # Check CPU usage
            if system_metrics.cpu_percent > 80:
                await self.alert_manager.create_alert(
                    self.alert_manager.rules["high_cpu_usage"],
                    system_metrics.cpu_percent
                )
            
            # Check memory usage
            if system_metrics.memory_percent > 85:
                await self.alert_manager.create_alert(
                    self.alert_manager.rules["high_memory_usage"],
                    system_metrics.memory_percent
                )
            
            # Check disk usage
            if system_metrics.disk_percent > 90:
                await self.alert_manager.create_alert(
                    self.alert_manager.rules["high_disk_usage"],
                    system_metrics.disk_percent
                )
                
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    async def _update_custom_metrics(self) -> None:
        """Update custom application metrics."""
        try:
            # Update request rate (example)
            current_time = time.time()
            if hasattr(self, '_last_request_count'):
                time_diff = current_time - self._last_request_time
                if time_diff > 0:
                    request_rate = (self._current_request_count - self._last_request_count) / time_diff
                    await self.metrics_collector.set_gauge("requests_per_second", request_rate)
                
                self._last_request_count = self._current_request_count
                self._last_request_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating custom metrics: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        try:
            metrics = await self.metrics_collector.collect_metrics()
            system_metrics = await self.system_monitor.get_system_metrics()
            
            return {
                "application_metrics": metrics,
                "system_metrics": system_metrics.dict(),
                "monitoring_status": {
                    "is_running": self.is_running,
                    "start_time": getattr(self, '_start_time', None),
                    "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            raise ServiceUnavailableError("Failed to retrieve metrics")
    
    async def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        try:
            metrics = await self.metrics_collector.collect_metrics()
            system_metrics = await self.system_monitor.get_system_metrics()
            
            prometheus_lines = []
            
            # Application metrics
            for metric_name, metric_data in metrics.items():
                if metric_data["latest_value"]:
                    value = metric_data["latest_value"]["value"]
                    labels = metric_data["latest_value"]["labels"]
                    
                    if labels:
                        label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
                        prometheus_lines.append(f'{metric_name}{{{label_str}}} {value}')
                    else:
                        prometheus_lines.append(f'{metric_name} {value}')
            
            # System metrics
            prometheus_lines.extend([
                f'system_cpu_percent {system_metrics.cpu_percent}',
                f'system_memory_percent {system_metrics.memory_percent}',
                f'system_memory_bytes {system_metrics.memory_used}',
                f'system_disk_percent {system_metrics.disk_percent}',
                f'system_disk_free_bytes {system_metrics.disk_free}',
                f'system_network_bytes_sent {system_metrics.network_bytes_sent}',
                f'system_network_bytes_recv {system_metrics.network_bytes_recv}'
            ])
            
            return "\n".join(prometheus_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating Prometheus metrics: {e}")
            raise ServiceUnavailableError("Failed to generate Prometheus metrics")
    
    async def get_performance_profiles(self) -> List[PerformanceProfile]:
        """Get all performance profiles."""
        return await self.performance_profiler.get_all_profiles()
    
    async def get_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get current alerts."""
        return await self.alert_manager.get_alerts(severity=severity)
    
    async def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        try:
            await self.stop()
            
            # Clear metrics and profiles
            await self.performance_profiler.clear_profiles()
            await self.alert_manager.clear_old_alerts(days=0)
            
            self.logger.info("Performance monitor cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_performance_monitor(config: MonitoringConfig) -> PerformanceMonitor:
    """Create and configure performance monitor."""
    return PerformanceMonitor(config)


def create_metrics_collector(config: MonitoringConfig) -> MetricsCollector:
    """Create and configure metrics collector."""
    return InMemoryMetricsCollector(config)


def create_system_monitor(config: MonitoringConfig) -> SystemMonitor:
    """Create and configure system monitor."""
    return SystemMetricsMonitor(config)


def create_performance_profiler(config: MonitoringConfig) -> PerformanceProfiler:
    """Create and configure performance profiler."""
    return PerformanceProfilerImpl(config)


def create_alert_manager(config: MonitoringConfig) -> AlertManager:
    """Create and configure alert manager."""
    return AlertManagerImpl(config)


# ============================================================================
# DECORATOR FOR FUNCTION PROFILING
# ============================================================================

def profile_function(monitor: Optional[PerformanceMonitor] = None):
    """Decorator to profile function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if monitor and monitor.performance_profiler:
                return await monitor.performance_profiler.profile_function(func)(*args, **kwargs)
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if monitor and monitor.performance_profiler:
                return monitor.performance_profiler.profile_function(func)(*args, **kwargs)
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
