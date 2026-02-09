"""
Monitoring and observability system for Export IA.
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores system metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self._lock = asyncio.Lock()
    
    async def record_metric(self, metric: Metric):
        """Record a metric."""
        async with self._lock:
            self.metrics[metric.name].append(metric)
    
    async def record_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Record a counter metric."""
        metric = Metric(name=name, value=value, tags=tags or {}, unit="count")
        await self.record_metric(metric)
    
    async def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric."""
        metric = Metric(name=name, value=value, tags=tags or {}, unit="gauge")
        await self.record_metric(metric)
    
    async def record_timing(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        metric = Metric(name=name, value=duration, tags=tags or {}, unit="seconds")
        await self.record_metric(metric)
    
    async def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[Metric]:
        """Get metrics for a specific name."""
        async with self._lock:
            if name not in self.metrics:
                return []
            
            metrics = list(self.metrics[name])
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics
    
    async def get_summary(self, name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        metrics = await self.get_metrics(name, since)
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
            "latest": values[-1] if values else 0
        }


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheck] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func
    
    async def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status="unhealthy",
                message="Check not registered"
            )
        
        try:
            result = await self.checks[name]()
            if isinstance(result, HealthCheck):
                self.results[name] = result
                return result
            else:
                # Simple boolean result
                status = "healthy" if result else "unhealthy"
                health_check = HealthCheck(
                    name=name,
                    status=status,
                    message="Check completed"
                )
                self.results[name] = health_check
                return health_check
        except Exception as e:
            health_check = HealthCheck(
                name=name,
                status="unhealthy",
                message=f"Check failed: {str(e)}"
            )
            self.results[name] = health_check
            return health_check
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        tasks = [self.run_check(name) for name in self.checks.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            name: result for name, result in zip(self.checks.keys(), results)
            if isinstance(result, HealthCheck)
        }
    
    def get_overall_status(self) -> str:
        """Get overall system health status."""
        if not self.results:
            return "unknown"
        
        statuses = [check.status for check in self.results.values()]
        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        else:
            return "healthy"


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start system monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.metrics_collector.record_gauge("system.cpu.usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            await self.metrics_collector.record_gauge("system.memory.usage", memory.percent)
            await self.metrics_collector.record_gauge("system.memory.available", memory.available)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            await self.metrics_collector.record_gauge("system.disk.usage", disk.percent)
            await self.metrics_collector.record_gauge("system.disk.free", disk.free)
            
            # Process count
            process_count = len(psutil.pids())
            await self.metrics_collector.record_gauge("system.processes.count", process_count)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class MonitoringManager:
    """Centralized monitoring management for Export IA."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self._initialized = False
    
    async def initialize(self):
        """Initialize monitoring system."""
        if self._initialized:
            return
        
        # Register default health checks
        self.health_checker.register_check("system", self._system_health_check)
        self.health_checker.register_check("memory", self._memory_health_check)
        self.health_checker.register_check("disk", self._disk_health_check)
        
        # Start system monitoring
        await self.system_monitor.start_monitoring()
        
        self._initialized = True
        logger.info("Monitoring system initialized")
    
    async def shutdown(self):
        """Shutdown monitoring system."""
        if not self._initialized:
            return
        
        await self.system_monitor.stop_monitoring()
        self._initialized = False
        logger.info("Monitoring system shutdown")
    
    async def _system_health_check(self) -> HealthCheck:
        """System health check."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 90:
                status = "degraded"
                message = f"High CPU usage: {cpu_percent}%"
            elif memory.percent > 90:
                status = "degraded"
                message = f"High memory usage: {memory.percent}%"
            else:
                status = "healthy"
                message = "System resources normal"
            
            return HealthCheck(
                name="system",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent
                }
            )
        except Exception as e:
            return HealthCheck(
                name="system",
                status="unhealthy",
                message=f"System check failed: {str(e)}"
            )
    
    async def _memory_health_check(self) -> HealthCheck:
        """Memory health check."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                status = "unhealthy"
                message = f"Critical memory usage: {memory.percent}%"
            elif memory.percent > 85:
                status = "degraded"
                message = f"High memory usage: {memory.percent}%"
            else:
                status = "healthy"
                message = f"Memory usage normal: {memory.percent}%"
            
            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                details={
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                }
            )
        except Exception as e:
            return HealthCheck(
                name="memory",
                status="unhealthy",
                message=f"Memory check failed: {str(e)}"
            )
    
    async def _disk_health_check(self) -> HealthCheck:
        """Disk health check."""
        try:
            disk = psutil.disk_usage('/')
            
            if disk.percent > 95:
                status = "unhealthy"
                message = f"Critical disk usage: {disk.percent}%"
            elif disk.percent > 85:
                status = "degraded"
                message = f"High disk usage: {disk.percent}%"
            else:
                status = "healthy"
                message = f"Disk usage normal: {disk.percent}%"
            
            return HealthCheck(
                name="disk",
                status=status,
                message=message,
                details={
                    "total": disk.total,
                    "free": disk.free,
                    "percent": disk.percent
                }
            )
        except Exception as e:
            return HealthCheck(
                name="disk",
                status="unhealthy",
                message=f"Disk check failed: {str(e)}"
            )
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        # Get all metric names
        metric_names = list(self.metrics_collector.metrics.keys())
        
        for name in metric_names:
            summary[name] = await self.metrics_collector.get_summary(name)
        
        return summary
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        await self.health_checker.run_all_checks()
        
        return {
            "overall_status": self.health_checker.get_overall_status(),
            "checks": {
                name: {
                    "status": check.status,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                    "details": check.details
                }
                for name, check in self.health_checker.results.items()
            }
        }


# Global monitoring manager instance
_monitoring_manager: Optional[MonitoringManager] = None


def get_monitoring_manager() -> MonitoringManager:
    """Get the global monitoring manager instance."""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager()
    return _monitoring_manager




