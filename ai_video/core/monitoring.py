from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import asyncio
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import statistics
from collections import defaultdict, deque
import weakref
    import psutil
    import prometheus_client as prometheus
from typing import Any, List, Dict, Optional
"""
AI Video System - Monitoring Module

Production-ready monitoring and observability including metrics collection,
health checks, alerting, and performance monitoring.
"""


try:
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Represents a health check result."""
    name: str
    status: str  # healthy, unhealthy, degraded
    message: str
    timestamp: datetime
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents a monitoring alert."""
    id: str
    severity: str  # info, warning, error, critical
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """
    Collects and manages system metrics.
    
    Features:
    - Custom metrics
    - System metrics
    - Prometheus integration
    - Metric aggregation
    """
    
    def __init__(self, retention_hours: int = 24):
        
    """__init__ function."""
self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.lock = threading.RLock()
        self.prometheus_metrics: Dict[str, Any] = {}
        
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        self.prometheus_metrics = {
            'counter': prometheus.Counter(
                'ai_video_operations_total',
                'Total number of operations',
                ['operation', 'status']
            ),
            'histogram': prometheus.Histogram(
                'ai_video_operation_duration_seconds',
                'Operation duration in seconds',
                ['operation'],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
            ),
            'gauge': prometheus.Gauge(
                'ai_video_active_operations',
                'Number of active operations',
                ['operation']
            ),
            'system_memory': prometheus.Gauge(
                'ai_video_system_memory_bytes',
                'System memory usage in bytes'
            ),
            'system_cpu': prometheus.Gauge(
                'ai_video_system_cpu_percent',
                'System CPU usage percentage'
            )
        }
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a metric."""
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics[name].append(metric_point)
            
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE and name in self.prometheus_metrics:
                self.prometheus_metrics[name].set(value)
    
    def record_operation(
        self,
        operation: str,
        duration: float,
        status: str = 'success',
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record operation metrics."""
        # Record custom metrics
        self.record_metric(f"{operation}_duration", duration, labels)
        self.record_metric(f"{operation}_count", 1, labels)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics['counter'].labels(
                operation=operation, status=status
            ).inc()
            
            self.prometheus_metrics['histogram'].labels(
                operation=operation
            ).observe(duration)
    
    def get_metric_stats(
        self,
        name: str,
        window_minutes: int = 60
    ) -> Optional[Dict[str, float]]:
        """Get statistics for a metric."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self.lock:
            if name not in self.metrics:
                return None
            
            recent_metrics = [
                m.value for m in self.metrics[name]
                if m.timestamp > cutoff_time
            ]
            
            if not recent_metrics:
                return None
            
            return {
                'count': len(recent_metrics),
                'min': min(recent_metrics),
                'max': max(recent_metrics),
                'avg': statistics.mean(recent_metrics),
                'median': statistics.median(recent_metrics),
                'std': statistics.stdev(recent_metrics) if len(recent_metrics) > 1 else 0
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not PSUTIL_AVAILABLE:
            return {'error': 'psutil not available'}
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            metrics = {
                'memory': {
                    'rss': memory_info.rss,
                    'vms': memory_info.vms,
                    'percent': process.memory_percent()
                },
                'cpu': {
                    'percent': process.cpu_percent(),
                    'system_percent': psutil.cpu_percent(interval=1)
                },
                'disk': {
                    'usage': psutil.disk_usage('/').percent
                },
                'threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                self.prometheus_metrics['system_memory'].set(memory_info.rss)
                self.prometheus_metrics['system_cpu'].set(process.cpu_percent())
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {'error': str(e)}
    
    def cleanup_old_metrics(self) -> int:
        """Remove old metrics."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        removed_count = 0
        
        with self.lock:
            for name in list(self.metrics.keys()):
                original_count = len(self.metrics[name])
                self.metrics[name] = deque(
                    [m for m in self.metrics[name] if m.timestamp > cutoff_time],
                    maxlen=10000
                )
                removed_count += original_count - len(self.metrics[name])
        
        return removed_count


class HealthChecker:
    """
    Performs health checks on system components.
    
    Features:
    - Custom health checks
    - System health checks
    - Health check scheduling
    - Health status aggregation
    """
    
    def __init__(self) -> Any:
        self.health_checks: Dict[str, Callable] = {}
        self.health_results: Dict[str, HealthCheck] = {}
        self.lock = threading.RLock()
        self.check_interval = 60  # seconds
        self._monitoring_task: Optional[asyncio.Task] = None
    
    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], Union[bool, Dict[str, Any]]]
    ) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    async def run_health_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.health_checks:
            raise ValueError(f"Health check '{name}' not found")
        
        start_time = time.time()
        
        try:
            result = self.health_checks[name]()
            duration = time.time() - start_time
            
            if isinstance(result, bool):
                status = 'healthy' if result else 'unhealthy'
                message = f"Health check {'passed' if result else 'failed'}"
                metadata = {}
            else:
                status = result.get('status', 'unhealthy')
                message = result.get('message', 'Health check completed')
                metadata = result.get('metadata', {})
            
            health_check = HealthCheck(
                name=name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration=duration,
                metadata=metadata
            )
            
            with self.lock:
                self.health_results[name] = health_check
            
            return health_check
            
        except Exception as e:
            duration = time.time() - start_time
            health_check = HealthCheck(
                name=name,
                status='unhealthy',
                message=f"Health check error: {str(e)}",
                timestamp=datetime.now(),
                duration=duration,
                metadata={'error': str(e)}
            )
            
            with self.lock:
                self.health_results[name] = health_check
            
            return health_check
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        tasks = [
            self.run_health_check(name)
            for name in self.health_checks.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_checks = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                name = list(self.health_checks.keys())[i]
                health_checks[name] = HealthCheck(
                    name=name,
                    status='unhealthy',
                    message=f"Health check failed: {str(result)}",
                    timestamp=datetime.now(),
                    duration=0.0,
                    metadata={'error': str(result)}
                )
            else:
                health_checks[result.name] = result
        
        return health_checks
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self.lock:
            if not self.health_results:
                return {
                    'status': 'unknown',
                    'message': 'No health checks performed',
                    'checks': {}
                }
            
            healthy_count = sum(
                1 for check in self.health_results.values()
                if check.status == 'healthy'
            )
            total_count = len(self.health_results)
            
            if healthy_count == total_count:
                status = 'healthy'
                message = 'All health checks passed'
            elif healthy_count > 0:
                status = 'degraded'
                message = f'{healthy_count}/{total_count} health checks passed'
            else:
                status = 'unhealthy'
                message = 'All health checks failed'
            
            return {
                'status': status,
                'message': message,
                'healthy_count': healthy_count,
                'total_count': total_count,
                'checks': {
                    name: {
                        'status': check.status,
                        'message': check.message,
                        'timestamp': check.timestamp.isoformat(),
                        'duration': check.duration
                    }
                    for name, check in self.health_results.items()
                }
            }
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring_task:
            return
        
        async def monitoring_loop():
            
    """monitoring_loop function."""
while True:
                try:
                    await self.run_all_health_checks()
                    await asyncio.sleep(self.check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(self.check_interval)
        
        self._monitoring_task = asyncio.create_task(monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None


class AlertManager:
    """
    Manages monitoring alerts and notifications.
    
    Features:
    - Alert creation and management
    - Alert severity levels
    - Alert resolution
    - Alert history
    """
    
    def __init__(self) -> Any:
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.lock = threading.RLock()
    
    def create_alert(
        self,
        severity: str,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new alert."""
        alert_id = f"alert_{int(time.time())}_{len(self.alerts)}"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.alerts[alert_id] = alert
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        logger.warning(f"Alert created: {severity.upper()} - {title}")
        return alert_id
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = datetime.now()
                return True
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self.lock:
            return [
                alert for alert in self.alerts.values()
                if not alert.resolved
            ]
    
    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """Get alerts by severity level."""
        with self.lock:
            return [
                alert for alert in self.alerts.values()
                if alert.severity == severity
            ]
    
    def cleanup_old_alerts(self, days: int = 30) -> int:
        """Remove old resolved alerts."""
        cutoff_time = datetime.now() - timedelta(days=days)
        alerts_to_remove = []
        
        with self.lock:
            for alert_id, alert in self.alerts.items():
                if (alert.resolved and alert.resolved_at and
                    alert.resolved_at < cutoff_time):
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.alerts[alert_id]
        
        return len(alerts_to_remove)
    
    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register an alert handler."""
        self.alert_handlers.append(handler)


class MonitoringDashboard:
    """
    Provides monitoring data for dashboards and APIs.
    
    Features:
    - Metrics aggregation
    - Health status
    - Alert summary
    - Performance trends
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        health_checker: HealthChecker,
        alert_manager: AlertManager
    ):
        
    """__init__ function."""
self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self.alert_manager = alert_manager
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'metrics': self.metrics_collector.get_system_metrics(),
                'health': self.health_checker.get_overall_health()
            },
            'alerts': {
                'active_count': len(self.alert_manager.get_active_alerts()),
                'by_severity': {
                    severity: len(self.alert_manager.get_alerts_by_severity(severity))
                    for severity in ['info', 'warning', 'error', 'critical']
                }
            },
            'performance': {
                'operation_stats': self._get_operation_stats(),
                'system_stats': self._get_system_stats()
            }
        }
    
    def _get_operation_stats(self) -> Dict[str, Any]:
        """Get operation performance statistics."""
        operations = ['video_generation', 'plugin_execution', 'file_processing']
        stats = {}
        
        for operation in operations:
            operation_stats = self.metrics_collector.get_metric_stats(
                f"{operation}_duration"
            )
            if operation_stats:
                stats[operation] = operation_stats
        
        return stats
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        return {
            'memory_usage': self.metrics_collector.get_metric_stats('system_memory'),
            'cpu_usage': self.metrics_collector.get_metric_stats('system_cpu'),
            'disk_usage': self.metrics_collector.get_metric_stats('system_disk')
        }
    
    def get_metrics_export(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        if format == 'json':
            return json.dumps(self.get_dashboard_data(), indent=2)
        elif format == 'prometheus' and PROMETHEUS_AVAILABLE:
            return prometheus.generate_latest()
        else:
            raise ValueError(f"Unsupported format: {format}")


# Built-in health checks
def check_system_resources() -> Dict[str, Any]:
    """Check system resource availability."""
    if not PSUTIL_AVAILABLE:
        return {
            'status': 'unhealthy',
            'message': 'psutil not available for system checks',
            'metadata': {}
        }
    
    try:
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            return {
                'status': 'unhealthy',
                'message': f'High memory usage: {memory.percent:.1f}%',
                'metadata': {'memory_percent': memory.percent}
            }
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            return {
                'status': 'unhealthy',
                'message': f'High disk usage: {disk.percent:.1f}%',
                'metadata': {'disk_percent': disk.percent}
            }
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            return {
                'status': 'degraded',
                'message': f'High CPU usage: {cpu_percent:.1f}%',
                'metadata': {'cpu_percent': cpu_percent}
            }
        
        return {
            'status': 'healthy',
            'message': 'System resources are normal',
            'metadata': {
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'cpu_percent': cpu_percent
            }
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'System check failed: {str(e)}',
            'metadata': {'error': str(e)}
        }


def check_database_connection() -> Dict[str, Any]:
    """Check database connectivity."""
    # This is a placeholder - implement actual database check
    try:
        # Add actual database connection check here
        return {
            'status': 'healthy',
            'message': 'Database connection is normal',
            'metadata': {}
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'Database connection failed: {str(e)}',
            'metadata': {'error': str(e)}
        }


# Global monitoring instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker()
alert_manager = AlertManager()
monitoring_dashboard = MonitoringDashboard(
    metrics_collector, health_checker, alert_manager
)

# Register default health checks
health_checker.register_health_check('system_resources', check_system_resources)
health_checker.register_health_check('database_connection', check_database_connection)


# Monitoring decorators
def monitor_operation(operation_name: str):
    """Decorator to monitor operation performance."""
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_operation(operation_name, duration, 'success')
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_operation(operation_name, duration, 'error')
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_operation(operation_name, duration, 'success')
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_operation(operation_name, duration, 'error')
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def alert_on_error(severity: str = 'error', source: str = 'unknown'):
    """Decorator to create alerts on errors."""
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                alert_manager.create_alert(
                    severity=severity,
                    title=f"Error in {func.__name__}",
                    message=str(e),
                    source=source,
                    metadata={'function': func.__name__, 'error': str(e)}
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                alert_manager.create_alert(
                    severity=severity,
                    title=f"Error in {func.__name__}",
                    message=str(e),
                    source=source,
                    metadata={'function': func.__name__, 'error': str(e)}
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


async def start_monitoring() -> None:
    """Start all monitoring services."""
    # Start health monitoring
    await health_checker.start_monitoring()
    
    # Start metrics collection
    asyncio.create_task(_collect_system_metrics())
    
    logger.info("Monitoring services started")


async def _collect_system_metrics() -> None:
    """Collect system metrics periodically."""
    while True:
        try:
            system_metrics = metrics_collector.get_system_metrics()
            
            if 'error' not in system_metrics:
                metrics_collector.record_metric(
                    'system_memory_percent',
                    system_metrics['memory']['percent']
                )
                metrics_collector.record_metric(
                    'system_cpu_percent',
                    system_metrics['cpu']['system_percent']
                )
                metrics_collector.record_metric(
                    'system_disk_percent',
                    system_metrics['disk']['usage']
                )
            
            await asyncio.sleep(30)  # Collect every 30 seconds
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
            await asyncio.sleep(30)


async def cleanup_monitoring_resources() -> None:
    """Cleanup monitoring resources."""
    # Stop health monitoring
    await health_checker.stop_monitoring()
    
    # Cleanup old metrics
    removed_metrics = metrics_collector.cleanup_old_metrics()
    
    # Cleanup old alerts
    removed_alerts = alert_manager.cleanup_old_alerts()
    
    logger.info(f"Monitoring cleanup: {removed_metrics} metrics, {removed_alerts} alerts removed") 