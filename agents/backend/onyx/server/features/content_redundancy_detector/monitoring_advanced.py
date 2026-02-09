"""
Advanced monitoring system for enterprise-grade observability
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Metric data"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert data"""
    id: str
    name: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check data"""
    name: str
    status: HealthStatus
    message: str
    response_time: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profile data"""
    name: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    response_times: List[float]
    throughput: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)


class AdvancedMonitoringSystem:
    """Advanced monitoring system"""
    
    def __init__(self):
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._alerts: Dict[str, Alert] = {}
        self._health_checks: Dict[str, HealthCheck] = {}
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._health_check_handlers: Dict[str, Callable] = {}
        self._performance_profiles: Dict[str, PerformanceProfile] = {}
        self._is_monitoring = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._system_metrics_interval = 30  # seconds
        self._health_check_interval = 60  # seconds
    
    async def start_monitoring(self) -> None:
        """Start monitoring system"""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Advanced monitoring system started")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring system"""
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced monitoring system stopped")
    
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        self._metrics[name].append(metric)
        
        # Check for alerts
        self._check_alert_rules(name, value)
    
    def increment_counter(self, name: str, value: int = 1, 
                         labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        # Get current counter value
        current_value = 0
        if name in self._metrics and self._metrics[name]:
            current_value = self._metrics[name][-1].value
        
        self.record_metric(name, current_value + value, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: Union[int, float],
                  labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric"""
        self.record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_histogram(self, name: str, value: Union[int, float],
                        labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric"""
        self.record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def add_alert_rule(self, rule_id: str, metric_name: str, threshold: float,
                      level: AlertLevel, message: str, 
                      condition: str = "greater_than") -> None:
        """Add alert rule"""
        self._alert_rules[rule_id] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "level": level,
            "message": message,
            "condition": condition,
            "enabled": True
        }
        
        logger.info(f"Alert rule added: {rule_id} for metric {metric_name}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove alert rule"""
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            logger.info(f"Alert rule removed: {rule_id}")
            return True
        return False
    
    def add_health_check(self, name: str, handler: Callable) -> None:
        """Add health check handler"""
        self._health_check_handlers[name] = handler
        logger.info(f"Health check added: {name}")
    
    def remove_health_check(self, name: str) -> bool:
        """Remove health check handler"""
        if name in self._health_check_handlers:
            del self._health_check_handlers[name]
            logger.info(f"Health check removed: {name}")
            return True
        return False
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add alert callback"""
        self._alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[Alert], None]) -> bool:
        """Remove alert callback"""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
            return True
        return False
    
    async def get_metric_history(self, name: str, duration_hours: int = 24) -> List[Metric]:
        """Get metric history"""
        if name not in self._metrics:
            return []
        
        cutoff_time = time.time() - (duration_hours * 3600)
        metrics = list(self._metrics[name])
        
        return [metric for metric in metrics if metric.timestamp > cutoff_time]
    
    async def get_metric_summary(self, name: str, duration_hours: int = 24) -> Optional[Dict[str, Any]]:
        """Get metric summary statistics"""
        metrics = await self.get_metric_history(name, duration_hours)
        
        if not metrics:
            return None
        
        values = [metric.value for metric in metrics]
        
        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "percentile_95": self._percentile(values, 95),
            "percentile_99": self._percentile(values, 99),
            "duration_hours": duration_hours
        }
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return [alert for alert in self._alerts.values() if not alert.resolved]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert"""
        if alert_id in self._alerts:
            alert = self._alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        health_checks = list(self._health_checks.values())
        
        if not health_checks:
            return {"status": "unknown", "message": "No health checks configured"}
        
        # Determine overall status
        statuses = [check.status for check in health_checks]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "total_checks": len(health_checks),
            "healthy_checks": len([c for c in health_checks if c.status == HealthStatus.HEALTHY]),
            "degraded_checks": len([c for c in health_checks if c.status == HealthStatus.DEGRADED]),
            "unhealthy_checks": len([c for c in health_checks if c.status == HealthStatus.UNHEALTHY]),
            "critical_checks": len([c for c in health_checks if c.status == HealthStatus.CRITICAL]),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time,
                    "timestamp": check.timestamp
                }
                for check in health_checks
            ]
        }
    
    async def get_performance_profile(self, name: str) -> Optional[PerformanceProfile]:
        """Get performance profile"""
        return self._performance_profiles.get(name)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            return {
                "timestamp": time.time(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": cpu_freq.current if cpu_freq else None,
                    "process_cpu": process_cpu
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free,
                    "process_memory": process_memory.rss
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._is_monitoring:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Run health checks
                await self._run_health_checks()
                
                # Update performance profiles
                await self._update_performance_profiles()
                
                # Wait for next interval
                await asyncio.sleep(self._system_metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _collect_system_metrics(self) -> None:
        """Collect system metrics"""
        try:
            system_metrics = await self.get_system_metrics()
            
            # Record CPU metrics
            self.set_gauge("system_cpu_percent", system_metrics.get("cpu", {}).get("percent", 0))
            self.set_gauge("system_cpu_count", system_metrics.get("cpu", {}).get("count", 0))
            
            # Record memory metrics
            memory = system_metrics.get("memory", {})
            self.set_gauge("system_memory_percent", memory.get("percent", 0))
            self.set_gauge("system_memory_used", memory.get("used", 0))
            self.set_gauge("system_memory_available", memory.get("available", 0))
            
            # Record disk metrics
            disk = system_metrics.get("disk", {})
            self.set_gauge("system_disk_percent", disk.get("percent", 0))
            self.set_gauge("system_disk_used", disk.get("used", 0))
            self.set_gauge("system_disk_free", disk.get("free", 0))
            
            # Record network metrics
            network = system_metrics.get("network", {})
            self.set_gauge("system_network_bytes_sent", network.get("bytes_sent", 0))
            self.set_gauge("system_network_bytes_recv", network.get("bytes_recv", 0))
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _run_health_checks(self) -> None:
        """Run health checks"""
        for name, handler in self._health_check_handlers.items():
            try:
                start_time = time.time()
                
                # Run health check
                if asyncio.iscoroutinefunction(handler):
                    result = await handler()
                else:
                    result = handler()
                
                response_time = time.time() - start_time
                
                # Create health check record
                health_check = HealthCheck(
                    name=name,
                    status=result.get("status", HealthStatus.HEALTHY),
                    message=result.get("message", "OK"),
                    response_time=response_time,
                    metadata=result.get("metadata", {})
                )
                
                self._health_checks[name] = health_check
                
            except Exception as e:
                # Create failed health check record
                health_check = HealthCheck(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {e}",
                    response_time=0,
                    metadata={"error": str(e)}
                )
                
                self._health_checks[name] = health_check
                logger.error(f"Health check failed for {name}: {e}")
    
    async def _update_performance_profiles(self) -> None:
        """Update performance profiles"""
        try:
            system_metrics = await self.get_system_metrics()
            
            # Get recent response times
            response_times = []
            if "api_response_time" in self._metrics:
                recent_metrics = await self.get_metric_history("api_response_time", 1)
                response_times = [metric.value for metric in recent_metrics]
            
            # Calculate throughput
            throughput = 0
            if "api_requests_total" in self._metrics:
                recent_metrics = await self.get_metric_history("api_requests_total", 1)
                if len(recent_metrics) >= 2:
                    throughput = recent_metrics[-1].value - recent_metrics[0].value
            
            # Calculate error rate
            error_rate = 0
            if "api_errors_total" in self._metrics and "api_requests_total" in self._metrics:
                error_metrics = await self.get_metric_history("api_errors_total", 1)
                request_metrics = await self.get_metric_history("api_requests_total", 1)
                
                if error_metrics and request_metrics:
                    error_rate = (error_metrics[-1].value / request_metrics[-1].value) * 100
            
            # Create performance profile
            profile = PerformanceProfile(
                name="system",
                cpu_usage=system_metrics.get("cpu", {}).get("percent", 0),
                memory_usage=system_metrics.get("memory", {}).get("percent", 0),
                disk_usage=system_metrics.get("disk", {}).get("percent", 0),
                network_io=system_metrics.get("network", {}),
                response_times=response_times,
                throughput=throughput,
                error_rate=error_rate
            )
            
            self._performance_profiles["system"] = profile
            
        except Exception as e:
            logger.error(f"Error updating performance profiles: {e}")
    
    def _check_alert_rules(self, metric_name: str, value: Union[int, float]) -> None:
        """Check alert rules for metric"""
        for rule_id, rule in self._alert_rules.items():
            if not rule.get("enabled", True):
                continue
            
            if rule["metric_name"] != metric_name:
                continue
            
            threshold = rule["threshold"]
            condition = rule.get("condition", "greater_than")
            
            # Check condition
            triggered = False
            if condition == "greater_than" and value > threshold:
                triggered = True
            elif condition == "less_than" and value < threshold:
                triggered = True
            elif condition == "equal" and value == threshold:
                triggered = True
            
            if triggered:
                # Check if alert already exists and is active
                existing_alert = None
                for alert in self._alerts.values():
                    if (alert.metric_name == metric_name and 
                        alert.threshold == threshold and 
                        not alert.resolved):
                        existing_alert = alert
                        break
                
                if not existing_alert:
                    # Create new alert
                    alert = Alert(
                        id=f"{rule_id}_{int(time.time())}",
                        name=rule_id,
                        level=rule["level"],
                        message=rule["message"],
                        metric_name=metric_name,
                        threshold=threshold,
                        current_value=value
                    )
                    
                    self._alerts[alert.id] = alert
                    
                    # Notify callbacks
                    for callback in self._alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")
                    
                    logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "is_monitoring": self._is_monitoring,
            "metrics_count": len(self._metrics),
            "total_metric_points": sum(len(metrics) for metrics in self._metrics.values()),
            "active_alerts": len([a for a in self._alerts.values() if not a.resolved]),
            "total_alerts": len(self._alerts),
            "health_checks": len(self._health_checks),
            "alert_rules": len(self._alert_rules),
            "performance_profiles": len(self._performance_profiles),
            "alert_callbacks": len(self._alert_callbacks)
        }


# Global monitoring system
monitoring_system = AdvancedMonitoringSystem()


# Helper functions
def record_metric(name: str, value: Union[int, float], 
                 metric_type: MetricType = MetricType.GAUGE,
                 labels: Optional[Dict[str, str]] = None) -> None:
    """Record a metric"""
    monitoring_system.record_metric(name, value, metric_type, labels)


def increment_counter(name: str, value: int = 1, 
                     labels: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter"""
    monitoring_system.increment_counter(name, value, labels)


def set_gauge(name: str, value: Union[int, float],
              labels: Optional[Dict[str, str]] = None) -> None:
    """Set a gauge"""
    monitoring_system.set_gauge(name, value, labels)


def record_histogram(name: str, value: Union[int, float],
                    labels: Optional[Dict[str, str]] = None) -> None:
    """Record a histogram"""
    monitoring_system.record_histogram(name, value, labels)


def add_alert_rule(rule_id: str, metric_name: str, threshold: float,
                  level: AlertLevel, message: str, 
                  condition: str = "greater_than") -> None:
    """Add alert rule"""
    monitoring_system.add_alert_rule(rule_id, metric_name, threshold, level, message, condition)


def add_health_check(name: str, handler: Callable) -> None:
    """Add health check"""
    monitoring_system.add_health_check(name, handler)


# Default health checks
def database_health_check() -> Dict[str, Any]:
    """Database health check"""
    try:
        # Simulate database check
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Database connection OK",
            "metadata": {"connection_pool": "active"}
        }
    except Exception as e:
        return {
            "status": HealthStatus.CRITICAL,
            "message": f"Database connection failed: {e}",
            "metadata": {"error": str(e)}
        }


def cache_health_check() -> Dict[str, Any]:
    """Cache health check"""
    try:
        # Simulate cache check
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Cache system OK",
            "metadata": {"cache_size": "normal"}
        }
    except Exception as e:
        return {
            "status": HealthStatus.DEGRADED,
            "message": f"Cache system degraded: {e}",
            "metadata": {"error": str(e)}
        }


def external_api_health_check() -> Dict[str, Any]:
    """External API health check"""
    try:
        # Simulate external API check
        return {
            "status": HealthStatus.HEALTHY,
            "message": "External APIs accessible",
            "metadata": {"response_time": "normal"}
        }
    except Exception as e:
        return {
            "status": HealthStatus.DEGRADED,
            "message": f"External API issues: {e}",
            "metadata": {"error": str(e)}
        }


# Initialize default health checks
monitoring_system.add_health_check("database", database_health_check)
monitoring_system.add_health_check("cache", cache_health_check)
monitoring_system.add_health_check("external_api", external_api_health_check)

# Initialize default alert rules
monitoring_system.add_alert_rule(
    "high_cpu_usage", "system_cpu_percent", 80.0, 
    AlertLevel.WARNING, "High CPU usage detected"
)

monitoring_system.add_alert_rule(
    "high_memory_usage", "system_memory_percent", 85.0, 
    AlertLevel.WARNING, "High memory usage detected"
)

monitoring_system.add_alert_rule(
    "high_disk_usage", "system_disk_percent", 90.0, 
    AlertLevel.ERROR, "High disk usage detected"
)

monitoring_system.add_alert_rule(
    "critical_cpu_usage", "system_cpu_percent", 95.0, 
    AlertLevel.CRITICAL, "Critical CPU usage detected"
)


