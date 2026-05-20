from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
from collections import defaultdict, deque
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge, Summary
import redis
import psycopg2
from contextlib import asynccontextmanager
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Production Monitoring System
Comprehensive monitoring, health checks, metrics, and observability for production.
"""



logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable
    interval: int = 30
    timeout: int = 10
    critical: bool = False
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNHEALTHY
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    max_failures: int = 3


@dataclass
class Metric:
    """Metric definition."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class SystemMetrics:
    """System-level metrics collection."""
    
    def __init__(self) -> Any:
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')
        self.network_io = Counter('system_network_bytes_total', 'Network I/O bytes', ['direction'])
        self.process_count = Gauge('system_process_count', 'Number of processes')
        
        self._last_network_stats = psutil.net_io_counters()
        self._last_network_check = time.time()
    
    def collect_system_metrics(self) -> Any:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.disk_usage.set((disk.used / disk.total) * 100)
            
            # Process count
            self.process_count.set(len(psutil.pids()))
            
            # Network I/O
            current_network = psutil.net_io_counters()
            current_time = time.time()
            
            if self._last_network_stats:
                time_diff = current_time - self._last_network_check
                bytes_sent = current_network.bytes_sent - self._last_network_stats.bytes_sent
                bytes_recv = current_network.bytes_recv - self._last_network_stats.bytes_recv
                
                self.network_io.labels(direction='sent').inc(bytes_sent)
                self.network_io.labels(direction='received').inc(bytes_recv)
            
            self._last_network_stats = current_network
            self._last_network_check = current_time
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class ApplicationMetrics:
    """Application-specific metrics."""
    
    def __init__(self) -> Any:
        # Operation metrics
        self.operation_requests = Counter('math_operation_requests_total', 'Total operation requests', ['operation_type', 'method'])
        self.operation_duration = Histogram('math_operation_duration_seconds', 'Operation duration', ['operation_type', 'method'])
        self.operation_errors = Counter('math_operation_errors_total', 'Total operation errors', ['operation_type', 'error_type'])
        
        # Cache metrics
        self.cache_hits = Counter('math_cache_hits_total', 'Total cache hits')
        self.cache_misses = Counter('math_cache_misses_total', 'Total cache misses')
        self.cache_size = Gauge('math_cache_size', 'Current cache size')
        self.cache_evictions = Counter('math_cache_evictions_total', 'Total cache evictions')
        
        # Workflow metrics
        self.workflow_executions = Counter('math_workflow_executions_total', 'Total workflow executions', ['workflow_name'])
        self.workflow_duration = Histogram('math_workflow_duration_seconds', 'Workflow duration', ['workflow_name'])
        self.workflow_errors = Counter('math_workflow_errors_total', 'Total workflow errors', ['workflow_name'])
        
        # API metrics
        self.api_requests = Counter('math_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status_code'])
        self.api_duration = Histogram('math_api_duration_seconds', 'API request duration', ['method', 'endpoint'])
        self.api_active_requests = Gauge('math_api_active_requests', 'Active API requests')
        
        # Performance metrics
        self.active_workers = Gauge('math_active_workers', 'Active worker threads')
        self.queue_size = Gauge('math_queue_size', 'Current queue size')
        self.memory_usage = Gauge('math_memory_usage_bytes', 'Application memory usage')
    
    def record_operation(self, operation_type: str, method: str, duration: float, success: bool, error_type: str = None):
        """Record operation metrics."""
        self.operation_requests.labels(operation_type=operation_type, method=method).inc()
        self.operation_duration.labels(operation_type=operation_type, method=method).observe(duration)
        
        if not success and error_type:
            self.operation_errors.labels(operation_type=operation_type, error_type=error_type).inc()
    
    def record_cache_operation(self, hit: bool, cache_size: int = None):
        """Record cache metrics."""
        if hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()
        
        if cache_size is not None:
            self.cache_size.set(cache_size)
    
    def record_workflow(self, workflow_name: str, duration: float, success: bool):
        """Record workflow metrics."""
        self.workflow_executions.labels(workflow_name=workflow_name).inc()
        self.workflow_duration.labels(workflow_name=workflow_name).observe(duration)
        
        if not success:
            self.workflow_errors.labels(workflow_name=workflow_name).inc()
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics."""
        self.api_requests.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.api_duration.labels(method=method, endpoint=endpoint).observe(duration)


class HealthChecker:
    """Health check system."""
    
    def __init__(self) -> Any:
        self.health_checks: Dict[str, HealthCheck] = {}
        self.overall_status = HealthStatus.HEALTHY
        self.last_check = datetime.now()
        self._running = False
        self._check_thread = None
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check."""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str):
        """Remove a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info(f"Removed health check: {name}")
    
    async def run_health_check(self, health_check: HealthCheck) -> HealthStatus:
        """Run a single health check."""
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(health_check.check_function):
                result = await asyncio.wait_for(health_check.check_function(), timeout=health_check.timeout)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, health_check.check_function
                )
            
            duration = time.time() - start_time
            
            if result:
                health_check.last_status = HealthStatus.HEALTHY
                health_check.consecutive_failures = 0
                health_check.last_error = None
                logger.debug(f"Health check {health_check.name} passed in {duration:.3f}s")
            else:
                health_check.last_status = HealthStatus.UNHEALTHY
                health_check.consecutive_failures += 1
                health_check.last_error = "Health check returned False"
                logger.warning(f"Health check {health_check.name} failed")
            
        except asyncio.TimeoutError:
            health_check.last_status = HealthStatus.UNHEALTHY
            health_check.consecutive_failures += 1
            health_check.last_error = f"Health check timed out after {health_check.timeout}s"
            logger.error(f"Health check {health_check.name} timed out")
            
        except Exception as e:
            health_check.last_status = HealthStatus.UNHEALTHY
            health_check.consecutive_failures += 1
            health_check.last_error = str(e)
            logger.error(f"Health check {health_check.name} failed with error: {e}")
        
        health_check.last_check = datetime.now()
        return health_check.last_status
    
    async def run_all_health_checks(self) -> Any:
        """Run all health checks."""
        tasks = []
        for health_check in self.health_checks.values():
            if (health_check.last_check is None or 
                datetime.now() - health_check.last_check >= timedelta(seconds=health_check.interval)):
                tasks.append(self.run_health_check(health_check))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self._update_overall_status()
    
    def _update_overall_status(self) -> Any:
        """Update overall health status."""
        critical_failures = 0
        total_failures = 0
        
        for health_check in self.health_checks.values():
            if health_check.last_status != HealthStatus.HEALTHY:
                total_failures += 1
                if health_check.critical:
                    critical_failures += 1
        
        if critical_failures > 0:
            self.overall_status = HealthStatus.CRITICAL
        elif total_failures > len(self.health_checks) * 0.5:
            self.overall_status = HealthStatus.UNHEALTHY
        elif total_failures > 0:
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.HEALTHY
        
        self.last_check = datetime.now()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        status = {
            "overall_status": self.overall_status.value,
            "last_check": self.last_check.isoformat(),
            "checks": {}
        }
        
        for name, health_check in self.health_checks.items():
            status["checks"][name] = {
                "status": health_check.last_status.value,
                "last_check": health_check.last_check.isoformat() if health_check.last_check else None,
                "last_error": health_check.last_error,
                "consecutive_failures": health_check.consecutive_failures,
                "critical": health_check.critical
            }
        
        return status
    
    def start_monitoring(self) -> Any:
        """Start health check monitoring."""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(target=self._monitor_loop, daemon=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        self._check_thread.start()
        logger.info("Health check monitoring started")
    
    def stop_monitoring(self) -> Any:
        """Stop health check monitoring."""
        self._running = False
        if self._check_thread:
            self._check_thread.join()
        logger.info("Health check monitoring stopped")
    
    def _monitor_loop(self) -> Any:
        """Health check monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self._running:
                loop.run_until_complete(self.run_all_health_checks())
                time.sleep(10)  # Check every 10 seconds
        finally:
            loop.close()


class PerformanceMonitor:
    """Performance monitoring and profiling."""
    
    def __init__(self, max_samples: int = 1000):
        
    """__init__ function."""
self.max_samples = max_samples
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.memory_usage: deque = deque(maxlen=max_samples)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.start_time = datetime.now()
    
    def record_operation_time(self, operation_type: str, duration: float):
        """Record operation execution time."""
        self.operation_times[operation_type].append(duration)
    
    def record_memory_usage(self, usage_bytes: int):
        """Record memory usage."""
        self.memory_usage.append(usage_bytes)
    
    def record_error(self, error_type: str):
        """Record error occurrence."""
        self.error_counts[error_type] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "operations": {},
            "memory": {},
            "errors": dict(self.error_counts)
        }
        
        # Operation statistics
        for operation_type, times in self.operation_times.items():
            if times:
                stats["operations"][operation_type] = {
                    "count": len(times),
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "min": min(times),
                    "max": max(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0
                }
        
        # Memory statistics
        if self.memory_usage:
            stats["memory"] = {
                "current": self.memory_usage[-1] if self.memory_usage else 0,
                "mean": statistics.mean(self.memory_usage),
                "max": max(self.memory_usage),
                "min": min(self.memory_usage)
            }
        
        return stats


class ProductionMonitoring:
    """Main production monitoring system."""
    
    def __init__(self, config) -> Any:
        self.config = config
        self.system_metrics = SystemMetrics()
        self.app_metrics = ApplicationMetrics()
        self.health_checker = HealthChecker()
        self.performance_monitor = PerformanceMonitor()
        
        # Setup health checks
        self._setup_health_checks()
        
        # Start monitoring
        self.health_checker.start_monitoring()
        
        logger.info("Production monitoring system initialized")
    
    def _setup_health_checks(self) -> Any:
        """Setup default health checks."""
        # System health checks
        self.health_checker.add_health_check(HealthCheck(
            name="system_cpu",
            check_function=self._check_cpu_usage,
            interval=60,
            critical=True
        ))
        
        self.health_checker.add_health_check(HealthCheck(
            name="system_memory",
            check_function=self._check_memory_usage,
            interval=60,
            critical=True
        ))
        
        self.health_checker.add_health_check(HealthCheck(
            name="system_disk",
            check_function=self._check_disk_usage,
            interval=300,
            critical=False
        ))
        
        # Database health check
        if self.config.database.host != "localhost":
            self.health_checker.add_health_check(HealthCheck(
                name="database_connection",
                check_function=self._check_database_connection,
                interval=30,
                critical=True
            ))
        
        # Redis health check
        if self.config.redis.host != "localhost":
            self.health_checker.add_health_check(HealthCheck(
                name="redis_connection",
                check_function=self._check_redis_connection,
                interval=30,
                critical=False
            ))
    
    def _check_cpu_usage(self) -> bool:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 90  # Alert if CPU usage > 90%
        except Exception as e:
            logger.error(f"CPU health check failed: {e}")
            return False
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if memory usage > 90%
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return False
    
    def _check_disk_usage(self) -> bool:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < 85  # Alert if disk usage > 85%
        except Exception as e:
            logger.error(f"Disk health check failed: {e}")
            return False
    
    def _check_database_connection(self) -> bool:
        """Check database connection."""
        try:
            conn = psycopg2.connect(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.name,
                user=self.config.database.user,
                password=self.config.database.password,
                connect_timeout=5
            )
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def _check_redis_connection(self) -> bool:
        """Check Redis connection."""
        try:
            r = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                password=self.config.redis.password,
                db=self.config.redis.db,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            r.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def collect_metrics(self) -> Any:
        """Collect all metrics."""
        # System metrics
        self.system_metrics.collect_system_metrics()
        
        # Application memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        self.app_metrics.memory_usage.set(memory_info.rss)
        
        # Performance monitoring
        self.performance_monitor.record_memory_usage(memory_info.rss)
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data."""
        return {
            "health": self.health_checker.get_health_status(),
            "performance": self.performance_monitor.get_performance_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    def shutdown(self) -> Any:
        """Shutdown monitoring system."""
        self.health_checker.stop_monitoring()
        logger.info("Production monitoring system shutdown")


# Context manager for monitoring
@asynccontextmanager
async def monitoring_context(config) -> Any:
    """Context manager for monitoring system."""
    monitoring = ProductionMonitoring(config)
    try:
        yield monitoring
    finally:
        monitoring.shutdown() 