"""
Advanced Monitoring System - Comprehensive Observability and Monitoring

This module provides advanced monitoring capabilities including:
- Real-time metrics collection and aggregation
- Distributed tracing and request tracking
- Advanced alerting and notification systems
- Performance monitoring and profiling
- Health checks and service discovery
- Log aggregation and analysis
- Dashboard and visualization
- Incident management and response
- Capacity planning and forecasting
- SLA monitoring and reporting
"""

import asyncio
import time
import uuid
import json
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import weakref
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
import statistics
import numpy as np
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"
    RATE = "rate"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class HealthStatus(Enum):
    """Health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class TraceType(Enum):
    """Trace types"""
    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    CACHE_OPERATION = "cache_operation"
    EXTERNAL_API = "external_api"
    BACKGROUND_JOB = "background_job"
    CUSTOM = "custom"

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.MEDIUM
    status: str = "active"
    source: str = ""
    metric_name: str = ""
    threshold: float = 0.0
    current_value: float = 0.0
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trace:
    """Distributed trace data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    operation_name: str = ""
    type: TraceType = TraceType.CUSTOM
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status: str = "success"
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheck:
    """Health check data structure"""
    name: str
    status: HealthStatus
    message: str = ""
    response_time: float = 0.0
    last_check: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Base classes
class BaseMonitor(ABC):
    """Base monitor class"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def collect_metrics(self) -> List[Metric]:
        """Collect metrics"""
        pass
    
    async def start_monitoring(self) -> None:
        """Start monitoring"""
        self.enabled = True
        logger.info(f"Started monitoring: {self.name}")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.enabled = False
        logger.info(f"Stopped monitoring: {self.name}")

# System Monitor
class SystemMonitor(BaseMonitor):
    """System resource monitor"""
    
    def __init__(self):
        super().__init__("system")
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.disk_history = deque(maxlen=100)
        self.network_history = deque(maxlen=100)
    
    async def collect_metrics(self) -> List[Metric]:
        """Collect system metrics"""
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        metrics.extend([
            Metric("cpu_usage_percent", cpu_percent, MetricType.GAUGE, {"core": "all"}),
            Metric("cpu_count", cpu_count, MetricType.GAUGE),
            Metric("cpu_frequency", cpu_freq.current if cpu_freq else 0, MetricType.GAUGE)
        ])
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        metrics.extend([
            Metric("memory_usage_percent", memory.percent, MetricType.GAUGE),
            Metric("memory_available_bytes", memory.available, MetricType.GAUGE),
            Metric("memory_used_bytes", memory.used, MetricType.GAUGE),
            Metric("memory_total_bytes", memory.total, MetricType.GAUGE),
            Metric("swap_usage_percent", swap.percent, MetricType.GAUGE),
            Metric("swap_used_bytes", swap.used, MetricType.GAUGE)
        ])
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics.extend([
            Metric("disk_usage_percent", disk.percent, MetricType.GAUGE),
            Metric("disk_free_bytes", disk.free, MetricType.GAUGE),
            Metric("disk_used_bytes", disk.used, MetricType.GAUGE),
            Metric("disk_total_bytes", disk.total, MetricType.GAUGE)
        ])
        
        if disk_io:
            metrics.extend([
                Metric("disk_read_bytes", disk_io.read_bytes, MetricType.COUNTER),
                Metric("disk_write_bytes", disk_io.write_bytes, MetricType.COUNTER),
                Metric("disk_read_count", disk_io.read_count, MetricType.COUNTER),
                Metric("disk_write_count", disk_io.write_count, MetricType.COUNTER)
            ])
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        if network_io:
            metrics.extend([
                Metric("network_bytes_sent", network_io.bytes_sent, MetricType.COUNTER),
                Metric("network_bytes_recv", network_io.bytes_recv, MetricType.COUNTER),
                Metric("network_packets_sent", network_io.packets_sent, MetricType.COUNTER),
                Metric("network_packets_recv", network_io.packets_recv, MetricType.COUNTER)
            ])
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        metrics.extend([
            Metric("process_memory_rss", process_memory.rss, MetricType.GAUGE),
            Metric("process_memory_vms", process_memory.vms, MetricType.GAUGE),
            Metric("process_cpu_percent", process.cpu_percent(), MetricType.GAUGE),
            Metric("process_num_threads", process.num_threads(), MetricType.GAUGE)
        ])
        
        # GC metrics
        gc_stats = gc.get_stats()
        metrics.append(Metric("gc_collections", sum(stat['collections'] for stat in gc_stats), MetricType.COUNTER))
        
        return metrics

# Application Monitor
class ApplicationMonitor(BaseMonitor):
    """Application-specific monitor"""
    
    def __init__(self):
        super().__init__("application")
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=1000)
        self.active_connections = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def collect_metrics(self) -> List[Metric]:
        """Collect application metrics"""
        metrics = []
        
        # Request metrics
        metrics.extend([
            Metric("requests_total", self.request_count, MetricType.COUNTER),
            Metric("errors_total", self.error_count, MetricType.COUNTER),
            Metric("active_connections", self.active_connections, MetricType.GAUGE)
        ])
        
        # Response time metrics
        if self.response_times:
            response_times = list(self.response_times)
            metrics.extend([
                Metric("response_time_avg", statistics.mean(response_times), MetricType.GAUGE),
                Metric("response_time_p50", statistics.median(response_times), MetricType.GAUGE),
                Metric("response_time_p95", np.percentile(response_times, 95), MetricType.GAUGE),
                Metric("response_time_p99", np.percentile(response_times, 99), MetricType.GAUGE),
                Metric("response_time_max", max(response_times), MetricType.GAUGE)
            ])
        
        # Cache metrics
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0
        
        metrics.extend([
            Metric("cache_hits_total", self.cache_hits, MetricType.COUNTER),
            Metric("cache_misses_total", self.cache_misses, MetricType.COUNTER),
            Metric("cache_hit_rate_percent", cache_hit_rate, MetricType.GAUGE)
        ])
        
        return metrics
    
    def record_request(self, response_time: float, is_error: bool = False) -> None:
        """Record request metrics"""
        self.request_count += 1
        if is_error:
            self.error_count += 1
        self.response_times.append(response_time)
    
    def record_cache_operation(self, hit: bool) -> None:
        """Record cache operation"""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

# Database Monitor
class DatabaseMonitor(BaseMonitor):
    """Database monitoring"""
    
    def __init__(self):
        super().__init__("database")
        self.query_count = 0
        self.slow_queries = 0
        self.connection_count = 0
        self.query_times = deque(maxlen=1000)
        self.connection_pool_size = 0
    
    async def collect_metrics(self) -> List[Metric]:
        """Collect database metrics"""
        metrics = []
        
        # Query metrics
        metrics.extend([
            Metric("queries_total", self.query_count, MetricType.COUNTER),
            Metric("slow_queries_total", self.slow_queries, MetricType.COUNTER),
            Metric("active_connections", self.connection_count, MetricType.GAUGE),
            Metric("connection_pool_size", self.connection_pool_size, MetricType.GAUGE)
        ])
        
        # Query time metrics
        if self.query_times:
            query_times = list(self.query_times)
            metrics.extend([
                Metric("query_time_avg", statistics.mean(query_times), MetricType.GAUGE),
                Metric("query_time_p95", np.percentile(query_times, 95), MetricType.GAUGE),
                Metric("query_time_max", max(query_times), MetricType.GAUGE)
            ])
        
        return metrics
    
    def record_query(self, query_time: float, is_slow: bool = False) -> None:
        """Record query metrics"""
        self.query_count += 1
        if is_slow:
            self.slow_queries += 1
        self.query_times.append(query_time)

# Metrics Collector
class MetricsCollector:
    """Centralized metrics collection system"""
    
    def __init__(self):
        self.monitors: List[BaseMonitor] = []
        self.metrics_storage: Dict[str, List[Metric]] = defaultdict(list)
        self.metrics_aggregation: Dict[str, Dict[str, Any]] = {}
        self.collection_interval = 10.0  # seconds
        self.retention_period = 24 * 60 * 60  # 24 hours in seconds
        self._collection_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def add_monitor(self, monitor: BaseMonitor) -> None:
        """Add monitor"""
        self.monitors.append(monitor)
        await monitor.start_monitoring()
        logger.info(f"Added monitor: {monitor.name}")
    
    async def remove_monitor(self, monitor: BaseMonitor) -> None:
        """Remove monitor"""
        if monitor in self.monitors:
            await monitor.stop_monitoring()
            self.monitors.remove(monitor)
            logger.info(f"Removed monitor: {monitor.name}")
    
    async def start_collection(self) -> None:
        """Start metrics collection"""
        if self._collection_task:
            return
        
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Started metrics collection")
    
    async def stop_collection(self) -> None:
        """Stop metrics collection"""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self) -> None:
        """Main collection loop"""
        while True:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old metrics"""
        while True:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_all_metrics(self) -> None:
        """Collect metrics from all monitors"""
        async with self._lock:
            for monitor in self.monitors:
                if monitor.enabled:
                    try:
                        metrics = await monitor.collect_metrics()
                        for metric in metrics:
                            self.metrics_storage[metric.name].append(metric)
                    except Exception as e:
                        logger.error(f"Error collecting metrics from {monitor.name}: {e}")
    
    async def _cleanup_old_metrics(self) -> None:
        """Cleanup old metrics"""
        async with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.retention_period)
            
            for metric_name, metrics in self.metrics_storage.items():
                self.metrics_storage[metric_name] = [
                    metric for metric in metrics
                    if metric.timestamp > cutoff_time
                ]
    
    async def get_metrics(self, 
                         metric_name: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 1000) -> List[Metric]:
        """Get metrics with filters"""
        async with self._lock:
            if metric_name:
                metrics = self.metrics_storage.get(metric_name, [])
            else:
                metrics = []
                for metric_list in self.metrics_storage.values():
                    metrics.extend(metric_list)
            
            # Apply time filters
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            # Sort by timestamp and limit
            metrics.sort(key=lambda x: x.timestamp, reverse=True)
            return metrics[:limit]
    
    async def get_aggregated_metrics(self, 
                                   metric_name: str,
                                   aggregation: str = "avg",
                                   time_window: int = 300) -> Dict[str, Any]:
        """Get aggregated metrics"""
        async with self._lock:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=time_window)
            
            metrics = await self.get_metrics(metric_name, start_time, end_time)
            
            if not metrics:
                return {"value": 0, "count": 0}
            
            values = [m.value for m in metrics]
            
            if aggregation == "avg":
                value = statistics.mean(values)
            elif aggregation == "sum":
                value = sum(values)
            elif aggregation == "min":
                value = min(values)
            elif aggregation == "max":
                value = max(values)
            elif aggregation == "p95":
                value = np.percentile(values, 95)
            elif aggregation == "p99":
                value = np.percentile(values, 99)
            else:
                value = statistics.mean(values)
            
            return {
                "value": value,
                "count": len(values),
                "time_window": time_window,
                "aggregation": aggregation
            }

# Alert Manager
class AlertManager:
    """Advanced alerting system"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[Callable] = []
        self.alert_history: deque = deque(maxlen=10000)
        self._lock = asyncio.Lock()
    
    async def add_alert_rule(self, 
                           name: str,
                           metric_name: str,
                           condition: str,
                           threshold: float,
                           severity: AlertSeverity = AlertSeverity.MEDIUM) -> None:
        """Add alert rule"""
        async with self._lock:
            self.alert_rules[name] = {
                "metric_name": metric_name,
                "condition": condition,
                "threshold": threshold,
                "severity": severity,
                "enabled": True,
                "created_at": datetime.utcnow()
            }
            logger.info(f"Added alert rule: {name}")
    
    async def remove_alert_rule(self, name: str) -> None:
        """Remove alert rule"""
        async with self._lock:
            if name in self.alert_rules:
                del self.alert_rules[name]
                logger.info(f"Removed alert rule: {name}")
    
    async def check_alerts(self, metrics: List[Metric]) -> List[Alert]:
        """Check metrics against alert rules"""
        triggered_alerts = []
        
        async with self._lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule["enabled"]:
                    continue
                
                # Find metrics matching the rule
                matching_metrics = [
                    m for m in metrics
                    if m.name == rule["metric_name"]
                ]
                
                for metric in matching_metrics:
                    if self._evaluate_condition(metric.value, rule["condition"], rule["threshold"]):
                        alert = Alert(
                            name=rule_name,
                            description=f"Alert triggered: {metric.name} {rule['condition']} {rule['threshold']}",
                            severity=rule["severity"],
                            source="alert_manager",
                            metric_name=metric.name,
                            threshold=rule["threshold"],
                            current_value=metric.value,
                            metadata={"rule": rule_name}
                        )
                        
                        triggered_alerts.append(alert)
                        self.alerts[alert.id] = alert
                        self.alert_history.append(alert)
                        
                        # Send notifications
                        await self._send_notifications(alert)
        
        return triggered_alerts
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return value == threshold
        elif condition == "not_equals":
            return value != threshold
        else:
            return False
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications"""
        for channel in self.notification_channels:
            try:
                await channel(alert)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
    
    def add_notification_channel(self, channel: Callable) -> None:
        """Add notification channel"""
        self.notification_channels.append(channel)
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert"""
        async with self._lock:
            alert = self.alerts.get(alert_id)
            if alert:
                alert.status = "resolved"
                alert.resolved_at = datetime.utcnow()
                return True
            return False
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        async with self._lock:
            return [alert for alert in self.alerts.values() if alert.status == "active"]

# Distributed Tracing
class TraceManager:
    """Distributed tracing system"""
    
    def __init__(self):
        self.traces: Dict[str, Trace] = {}
        self.trace_spans: Dict[str, List[Trace]] = defaultdict(list)
        self.trace_storage: deque = deque(maxlen=10000)
        self._lock = asyncio.Lock()
    
    async def start_trace(self, 
                         operation_name: str,
                         trace_type: TraceType = TraceType.CUSTOM,
                         parent_id: Optional[str] = None,
                         tags: Optional[Dict[str, str]] = None) -> Trace:
        """Start new trace"""
        trace = Trace(
            operation_name=operation_name,
            type=trace_type,
            parent_id=parent_id,
            tags=tags or {}
        )
        
        async with self._lock:
            self.traces[trace.id] = trace
            if parent_id:
                self.trace_spans[parent_id].append(trace)
        
        return trace
    
    async def finish_trace(self, trace_id: str, status: str = "success") -> None:
        """Finish trace"""
        async with self._lock:
            trace = self.traces.get(trace_id)
            if trace:
                trace.end_time = datetime.utcnow()
                trace.duration = (trace.end_time - trace.start_time).total_seconds()
                trace.status = status
                
                # Move to storage
                self.trace_storage.append(trace)
                
                # Remove from active traces
                del self.traces[trace_id]
    
    async def add_trace_log(self, trace_id: str, message: str, level: str = "info") -> None:
        """Add log to trace"""
        async with self._lock:
            trace = self.traces.get(trace_id)
            if trace:
                trace.logs.append({
                    "message": message,
                    "level": level,
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    async def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID"""
        async with self._lock:
            return self.traces.get(trace_id)
    
    async def get_traces_by_operation(self, operation_name: str, limit: int = 100) -> List[Trace]:
        """Get traces by operation name"""
        async with self._lock:
            traces = [
                trace for trace in self.trace_storage
                if trace.operation_name == operation_name
            ]
            return traces[-limit:]
    
    @asynccontextmanager
    async def trace_context(self, 
                           operation_name: str,
                           trace_type: TraceType = TraceType.CUSTOM,
                           parent_id: Optional[str] = None,
                           tags: Optional[Dict[str, str]] = None):
        """Context manager for tracing"""
        trace = await self.start_trace(operation_name, trace_type, parent_id, tags)
        try:
            yield trace
        except Exception as e:
            await self.add_trace_log(trace.id, f"Error: {str(e)}", "error")
            await self.finish_trace(trace.id, "error")
            raise
        else:
            await self.finish_trace(trace.id, "success")

# Health Check Manager
class HealthCheckManager:
    """Health check management system"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, HealthCheck] = {}
        self.check_interval = 30.0  # seconds
        self._check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def add_health_check(self, name: str, check_func: Callable) -> None:
        """Add health check"""
        async with self._lock:
            self.health_checks[name] = check_func
            logger.info(f"Added health check: {name}")
    
    async def remove_health_check(self, name: str) -> None:
        """Remove health check"""
        async with self._lock:
            if name in self.health_checks:
                del self.health_checks[name]
                self.health_status.pop(name, None)
                logger.info(f"Removed health check: {name}")
    
    async def start_health_checks(self) -> None:
        """Start health check monitoring"""
        if self._check_task:
            return
        
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started health check monitoring")
    
    async def stop_health_checks(self) -> None:
        """Stop health check monitoring"""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health check monitoring")
    
    async def _health_check_loop(self) -> None:
        """Health check monitoring loop"""
        while True:
            try:
                await self._run_all_health_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _run_all_health_checks(self) -> None:
        """Run all health checks"""
        async with self._lock:
            for name, check_func in self.health_checks.items():
                try:
                    start_time = time.time()
                    result = await check_func()
                    response_time = time.time() - start_time
                    
                    if isinstance(result, dict):
                        status = HealthStatus(result.get("status", "unknown"))
                        message = result.get("message", "")
                    else:
                        status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                        message = "Health check passed" if result else "Health check failed"
                    
                    self.health_status[name] = HealthCheck(
                        name=name,
                        status=status,
                        message=message,
                        response_time=response_time,
                        last_check=datetime.utcnow()
                    )
                    
                except Exception as e:
                    self.health_status[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check error: {str(e)}",
                        response_time=0.0,
                        last_check=datetime.utcnow()
                    )
    
    async def get_health_status(self) -> Dict[str, HealthCheck]:
        """Get health status of all checks"""
        async with self._lock:
            return self.health_status.copy()
    
    async def get_overall_health(self) -> HealthStatus:
        """Get overall system health"""
        async with self._lock:
            if not self.health_status:
                return HealthStatus.UNKNOWN
            
            statuses = [check.status for check in self.health_status.values()]
            
            if all(status == HealthStatus.HEALTHY for status in statuses):
                return HealthStatus.HEALTHY
            elif any(status == HealthStatus.UNHEALTHY for status in statuses):
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.DEGRADED

# Advanced Monitoring Manager
class AdvancedMonitoringManager:
    """Main advanced monitoring management system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.trace_manager = TraceManager()
        self.health_check_manager = HealthCheckManager()
        
        # Built-in monitors
        self.system_monitor = SystemMonitor()
        self.application_monitor = ApplicationMonitor()
        self.database_monitor = DatabaseMonitor()
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize monitoring system"""
        if self._initialized:
            return
        
        # Add built-in monitors
        await self.metrics_collector.add_monitor(self.system_monitor)
        await self.metrics_collector.add_monitor(self.application_monitor)
        await self.metrics_collector.add_monitor(self.database_monitor)
        
        # Start collection and health checks
        await self.metrics_collector.start_collection()
        await self.health_check_manager.start_health_checks()
        
        # Add default alert rules
        await self._setup_default_alerts()
        
        self._initialized = True
        logger.info("Advanced monitoring system initialized")
    
    async def _setup_default_alerts(self) -> None:
        """Setup default alert rules"""
        default_alerts = [
            ("high_cpu_usage", "cpu_usage_percent", "greater_than", 80.0, AlertSeverity.HIGH),
            ("high_memory_usage", "memory_usage_percent", "greater_than", 85.0, AlertSeverity.HIGH),
            ("high_disk_usage", "disk_usage_percent", "greater_than", 90.0, AlertSeverity.CRITICAL),
            ("high_error_rate", "errors_total", "greater_than", 10.0, AlertSeverity.MEDIUM),
            ("slow_response_time", "response_time_avg", "greater_than", 2.0, AlertSeverity.MEDIUM)
        ]
        
        for name, metric, condition, threshold, severity in default_alerts:
            await self.alert_manager.add_alert_rule(name, metric, condition, threshold, severity)
    
    async def shutdown(self) -> None:
        """Shutdown monitoring system"""
        await self.metrics_collector.stop_collection()
        await self.health_check_manager.stop_health_checks()
        self._initialized = False
        logger.info("Advanced monitoring system shut down")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring system summary"""
        return {
            "initialized": self._initialized,
            "monitors_count": len(self.metrics_collector.monitors),
            "metrics_count": len(self.metrics_collector.metrics_storage),
            "alert_rules_count": len(self.alert_manager.alert_rules),
            "active_alerts_count": len([a for a in self.alert_manager.alerts.values() if a.status == "active"]),
            "traces_count": len(self.trace_manager.traces),
            "health_checks_count": len(self.health_check_manager.health_checks),
            "collection_interval": self.metrics_collector.collection_interval,
            "retention_period": self.metrics_collector.retention_period
        }
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        # Get recent metrics
        recent_metrics = await self.metrics_collector.get_metrics(limit=100)
        
        # Get active alerts
        active_alerts = await self.alert_manager.get_active_alerts()
        
        # Get health status
        health_status = await self.health_check_manager.get_health_status()
        overall_health = await self.health_check_manager.get_overall_health()
        
        # Get aggregated metrics
        cpu_usage = await self.metrics_collector.get_aggregated_metrics("cpu_usage_percent", "avg", 300)
        memory_usage = await self.metrics_collector.get_aggregated_metrics("memory_usage_percent", "avg", 300)
        response_time = await self.metrics_collector.get_aggregated_metrics("response_time_avg", "avg", 300)
        
        return {
            "overall_health": overall_health.value,
            "health_checks": {name: {
                "status": check.status.value,
                "message": check.message,
                "response_time": check.response_time,
                "last_check": check.last_check.isoformat()
            } for name, check in health_status.items()},
            "active_alerts": [{
                "id": alert.id,
                "name": alert.name,
                "severity": alert.severity.value,
                "description": alert.description,
                "triggered_at": alert.triggered_at.isoformat()
            } for alert in active_alerts],
            "metrics": {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "response_time": response_time
            },
            "recent_metrics": [{
                "name": metric.name,
                "value": metric.value,
                "type": metric.type.value,
                "timestamp": metric.timestamp.isoformat(),
                "labels": metric.labels
            } for metric in recent_metrics[:20]]
        }

# Global monitoring manager instance
_global_monitoring_manager: Optional[AdvancedMonitoringManager] = None

def get_monitoring_manager() -> AdvancedMonitoringManager:
    """Get global monitoring manager instance"""
    global _global_monitoring_manager
    if _global_monitoring_manager is None:
        _global_monitoring_manager = AdvancedMonitoringManager()
    return _global_monitoring_manager

async def initialize_monitoring() -> None:
    """Initialize global monitoring system"""
    manager = get_monitoring_manager()
    await manager.initialize()

async def shutdown_monitoring() -> None:
    """Shutdown global monitoring system"""
    manager = get_monitoring_manager()
    await manager.shutdown()

async def get_dashboard_data() -> Dict[str, Any]:
    """Get dashboard data using global monitoring manager"""
    manager = get_monitoring_manager()
    return await manager.get_dashboard_data()





















