"""
Advanced monitoring and metrics system for Facebook Posts API
Real-time performance monitoring, health checks, and alerting
"""

import time
import asyncio
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import structlog

from ..core.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: str
    timestamp: datetime
    response_time: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: str  # "critical", "warning", "info"
    message: str
    cooldown: int = 300  # 5 minutes


class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            
            # Keep only recent values
            if len(self.histograms[key]) > self.max_points:
                self.histograms[key] = self.histograms[key][-self.max_points:]
    
    def record_timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        self.record_histogram(f"{name}_duration", duration, labels)
        self.increment_counter(f"{name}_count", labels=labels)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a key for labeled metrics"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: {
                        "count": len(values),
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "avg": sum(values) / len(values) if values else 0,
                        "p50": self._percentile(values, 50),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99)
                    }
                    for name, values in self.histograms.items()
                }
            }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheck] = {}
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Register a health check function"""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                response_time = time.time() - start_time
                
                # Update response time
                result.response_time = response_time
                result.timestamp = datetime.now()
                
                results[name] = result
                self.results[name] = result
                
            except Exception as e:
                logger.error("Health check failed", check=name, error=str(e))
                results[name] = HealthCheck(
                    name=name,
                    status="unhealthy",
                    message=f"Check failed: {str(e)}",
                    timestamp=datetime.now(),
                    response_time=0.0
                )
        
        return results
    
    def get_overall_status(self) -> str:
        """Get overall system health status"""
        if not self.results:
            return "unknown"
        
        statuses = [result.status for result in self.results.values()]
        
        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        else:
            return "healthy"


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.triggered_alerts: Dict[str, datetime] = {}
        self.notification_handlers: List[Callable] = []
    
    def register_alert(self, alert: Alert):
        """Register an alert"""
        self.alerts.append(alert)
    
    def register_notification_handler(self, handler: Callable[[Alert, Dict[str, Any]], None]):
        """Register a notification handler"""
        self.notification_handlers.append(handler)
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check all registered alerts"""
        for alert in self.alerts:
            try:
                # Check if alert is in cooldown
                last_triggered = self.triggered_alerts.get(alert.name)
                if last_triggered and (datetime.now() - last_triggered).seconds < alert.cooldown:
                    continue
                
                # Check alert condition
                if alert.condition(metrics):
                    self.triggered_alerts[alert.name] = datetime.now()
                    
                    # Send notifications
                    for handler in self.notification_handlers:
                        try:
                            await handler(alert, metrics) if asyncio.iscoroutinefunction(handler) else handler(alert, metrics)
                        except Exception as e:
                            logger.error("Notification handler failed", alert=alert.name, error=str(e))
                    
                    logger.warning("Alert triggered", alert=alert.name, message=alert.message)
                
            except Exception as e:
                logger.error("Alert check failed", alert=alert.name, error=str(e))


class SystemMonitor:
    """Main system monitoring class"""
    
    def __init__(self):
        self.settings = get_settings()
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        self.start_time = time.time()
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Register default alerts
        self._register_default_alerts()
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        
        def check_database():
            # Mock database check
            return HealthCheck(
                name="database",
                status="healthy",
                message="Database connection is healthy",
                timestamp=datetime.now(),
                response_time=0.0
            )
        
        def check_redis():
            # Mock Redis check
            return HealthCheck(
                name="redis",
                status="healthy",
                message="Redis connection is healthy",
                timestamp=datetime.now(),
                response_time=0.0
            )
        
        def check_ai_service():
            # Mock AI service check
            return HealthCheck(
                name="ai_service",
                status="healthy",
                message="AI service is responding",
                timestamp=datetime.now(),
                response_time=0.0
            )
        
        self.health_checker.register_check("database", check_database)
        self.health_checker.register_check("redis", check_redis)
        self.health_checker.register_check("ai_service", check_ai_service)
    
    def _register_default_alerts(self):
        """Register default alerts"""
        
        def high_cpu_alert(metrics):
            cpu_usage = metrics.get("gauges", {}).get("system_cpu_percent", 0)
            return cpu_usage > 80
        
        def high_memory_alert(metrics):
            memory_usage = metrics.get("gauges", {}).get("system_memory_percent", 0)
            return memory_usage > 85
        
        def high_error_rate_alert(metrics):
            error_count = metrics.get("counters", {}).get("api_errors_total", 0)
            total_requests = metrics.get("counters", {}).get("api_requests_total", 1)
            error_rate = error_count / total_requests if total_requests > 0 else 0
            return error_rate > 0.1  # 10% error rate
        
        def slow_response_alert(metrics):
            response_time = metrics.get("histograms", {}).get("api_request_duration", {})
            p95 = response_time.get("p95", 0)
            return p95 > 2.0  # 2 seconds
        
        self.alert_manager.register_alert(Alert(
            name="high_cpu",
            condition=high_cpu_alert,
            severity="warning",
            message="CPU usage is above 80%"
        ))
        
        self.alert_manager.register_alert(Alert(
            name="high_memory",
            condition=high_memory_alert,
            severity="warning",
            message="Memory usage is above 85%"
        ))
        
        self.alert_manager.register_alert(Alert(
            name="high_error_rate",
            condition=high_error_rate_alert,
            severity="critical",
            message="API error rate is above 10%"
        ))
        
        self.alert_manager.register_alert(Alert(
            name="slow_response",
            condition=slow_response_alert,
            severity="warning",
            message="API response time P95 is above 2 seconds"
        ))
    
    async def collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_collector.set_gauge("system_memory_percent", memory.percent)
            self.metrics_collector.set_gauge("system_memory_available", memory.available)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics_collector.set_gauge("system_disk_percent", disk.percent)
            self.metrics_collector.set_gauge("system_disk_free", disk.free)
            
            # Network I/O
            network = psutil.net_io_counters()
            self.metrics_collector.set_gauge("system_network_bytes_sent", network.bytes_sent)
            self.metrics_collector.set_gauge("system_network_bytes_recv", network.bytes_recv)
            
            # Process info
            process = psutil.Process()
            self.metrics_collector.set_gauge("process_memory_percent", process.memory_percent())
            self.metrics_collector.set_gauge("process_cpu_percent", process.cpu_percent())
            
            # Uptime
            uptime = time.time() - self.start_time
            self.metrics_collector.set_gauge("system_uptime", uptime)
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                await self.collect_system_metrics()
                
                # Run health checks
                health_results = await self.health_checker.run_checks()
                
                # Get all metrics
                metrics = self.metrics_collector.get_metrics()
                
                # Check alerts
                await self.alert_manager.check_alerts(metrics)
                
                # Log monitoring status
                logger.debug("Monitoring cycle completed", 
                           metrics_count=len(metrics.get("gauges", {})),
                           health_checks=len(health_results))
                
                # Wait before next cycle
                await asyncio.sleep(30)  # 30 seconds
                
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def start(self):
        """Start monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self.monitor_loop())
        logger.info("System monitoring started")
    
    async def stop(self):
        """Stop monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitoring stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics_collector.get_metrics()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        health_results = self.health_checker.results
        overall_status = self.health_checker.get_overall_status()
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
            "checks": {
                name: {
                    "status": result.status,
                    "message": result.message,
                    "response_time": result.response_time,
                    "timestamp": result.timestamp.isoformat(),
                    "details": result.details
                }
                for name, result in health_results.items()
            }
        }
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics"""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code)
        }
        
        # Record timing
        self.metrics_collector.record_timing("api_request", duration, labels)
        
        # Increment counters
        self.metrics_collector.increment_counter("api_requests_total", labels=labels)
        
        if status_code >= 400:
            self.metrics_collector.increment_counter("api_errors_total", labels=labels)
    
    def record_post_generation(self, duration: float, success: bool):
        """Record post generation metrics"""
        labels = {"success": str(success).lower()}
        
        self.metrics_collector.record_timing("post_generation", duration, labels)
        self.metrics_collector.increment_counter("posts_generated_total", labels=labels)
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation metrics"""
        labels = {
            "operation": operation,
            "hit": str(hit).lower()
        }
        
        self.metrics_collector.increment_counter("cache_operations_total", labels=labels)


# Global monitor instance
_monitor: Optional[SystemMonitor] = None


def get_monitor() -> SystemMonitor:
    """Get global monitor instance"""
    global _monitor
    
    if _monitor is None:
        _monitor = SystemMonitor()
    
    return _monitor


async def start_monitoring():
    """Start global monitoring"""
    monitor = get_monitor()
    await monitor.start()


async def stop_monitoring():
    """Stop global monitoring"""
    global _monitor
    
    if _monitor:
        await _monitor.stop()
        _monitor = None


# Context manager for timing operations
class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, monitor: SystemMonitor, operation: str, labels: Optional[Dict[str, str]] = None):
        self.monitor = monitor
        self.operation = operation
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.metrics_collector.record_timing(self.operation, duration, self.labels)


# Decorator for timing functions
def timed(operation: str, labels: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitor = get_monitor()
            with TimingContext(monitor, operation, labels):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            monitor = get_monitor()
            with TimingContext(monitor, operation, labels):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Export all classes and functions
__all__ = [
    # Data classes
    'MetricPoint',
    'HealthCheck',
    'Alert',
    
    # Main classes
    'MetricsCollector',
    'HealthChecker',
    'AlertManager',
    'SystemMonitor',
    
    # Utility functions
    'get_monitor',
    'start_monitoring',
    'stop_monitoring',
    'TimingContext',
    'timed',
]






























