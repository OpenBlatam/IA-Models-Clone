"""
Monitoring Service
==================

Advanced monitoring service for real-time system monitoring and alerting.
"""

from __future__ import annotations
import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import psutil
import threading
from collections import deque, defaultdict
import statistics

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution, measure_performance


logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert level enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertCondition(str, Enum):
    """Alert condition enumeration"""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUAL_TO = "eq"
    NOT_EQUAL_TO = "ne"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN_OR_EQUAL = "lte"


@dataclass
class Alert:
    """Alert representation"""
    id: str
    name: str
    description: str
    level: AlertLevel
    condition: AlertCondition
    threshold: float
    metric_name: str
    is_active: bool = True
    created_at: datetime = field(default_factory=DateTimeHelpers.now_utc)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    cooldown_seconds: int = 300  # 5 minutes


@dataclass
class AlertEvent:
    """Alert event representation"""
    alert_id: str
    alert_name: str
    level: AlertLevel
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=DateTimeHelpers.now_utc)
    resolved: bool = False


@dataclass
class Metric:
    """Metric representation"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=DateTimeHelpers.now_utc)


@dataclass
class SystemHealth:
    """System health representation"""
    status: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    response_time: float
    error_rate: float
    timestamp: datetime = field(default_factory=DateTimeHelpers.now_utc)


class MonitoringService:
    """Advanced monitoring service with real-time metrics and alerting"""
    
    def __init__(self):
        self._is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._alerts: Dict[str, Alert] = {}
        self._alert_events: List[AlertEvent] = []
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._health_history: deque = deque(maxlen=100)
        self._lock = asyncio.Lock()
        self._alert_handlers: List[Callable[[AlertEvent], None]] = []
        self._metric_handlers: List[Callable[[Metric], None]] = []
        self._initialize_default_alerts()
    
    def _initialize_default_alerts(self):
        """Initialize default system alerts"""
        default_alerts = [
            Alert(
                id="cpu_high",
                name="High CPU Usage",
                description="CPU usage is above 80%",
                level=AlertLevel.WARNING,
                condition=AlertCondition.GREATER_THAN,
                threshold=80.0,
                metric_name="system.cpu.percent"
            ),
            Alert(
                id="memory_high",
                name="High Memory Usage",
                description="Memory usage is above 85%",
                level=AlertLevel.WARNING,
                condition=AlertCondition.GREATER_THAN,
                threshold=85.0,
                metric_name="system.memory.percent"
            ),
            Alert(
                id="disk_high",
                name="High Disk Usage",
                description="Disk usage is above 90%",
                level=AlertLevel.CRITICAL,
                condition=AlertCondition.GREATER_THAN,
                threshold=90.0,
                metric_name="system.disk.percent"
            ),
            Alert(
                id="response_time_high",
                name="High Response Time",
                description="Average response time is above 1 second",
                level=AlertLevel.WARNING,
                condition=AlertCondition.GREATER_THAN,
                threshold=1000.0,
                metric_name="application.response_time"
            ),
            Alert(
                id="error_rate_high",
                name="High Error Rate",
                description="Error rate is above 5%",
                level=AlertLevel.ERROR,
                condition=AlertCondition.GREATER_THAN,
                threshold=5.0,
                metric_name="application.error_rate"
            )
        ]
        
        for alert in default_alerts:
            self._alerts[alert.id] = alert
    
    async def start(self):
        """Start the monitoring service"""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_worker())
        
        logger.info("Monitoring service started successfully")
    
    async def stop(self):
        """Stop the monitoring service"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring service stopped")
    
    async def _monitoring_worker(self):
        """Monitoring worker task"""
        while self._is_running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check alerts
                await self._check_alerts()
                
                # Update system health
                await self._update_system_health()
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
            
            except Exception as e:
                logger.error(f"Monitoring worker error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric("system.cpu.percent", cpu_percent, MetricType.GAUGE)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self._record_metric("system.memory.percent", memory.percent, MetricType.GAUGE)
            await self._record_metric("system.memory.available", memory.available, MetricType.GAUGE)
            await self._record_metric("system.memory.used", memory.used, MetricType.GAUGE)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self._record_metric("system.disk.percent", disk_percent, MetricType.GAUGE)
            await self._record_metric("system.disk.free", disk.free, MetricType.GAUGE)
            
            # Network metrics
            network_io = psutil.net_io_counters()
            await self._record_metric("system.network.bytes_sent", network_io.bytes_sent, MetricType.COUNTER)
            await self._record_metric("system.network.bytes_recv", network_io.bytes_recv, MetricType.COUNTER)
            
            # Process metrics
            process = psutil.Process()
            await self._record_metric("system.process.cpu_percent", process.cpu_percent(), MetricType.GAUGE)
            await self._record_metric("system.process.memory_percent", process.memory_percent(), MetricType.GAUGE)
            await self._record_metric("system.process.num_threads", process.num_threads(), MetricType.GAUGE)
        
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _record_metric(self, name: str, value: float, metric_type: MetricType, labels: Optional[Dict[str, str]] = None):
        """Record a metric"""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                labels=labels or {}
            )
            
            # Store metric
            self._metrics[name].append(metric)
            
            # Notify metric handlers
            for handler in self._metric_handlers:
                try:
                    handler(metric)
                except Exception as e:
                    logger.error(f"Metric handler error: {e}")
        
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
    
    async def _check_alerts(self):
        """Check alert conditions"""
        try:
            for alert in self._alerts.values():
                if not alert.is_active:
                    continue
                
                # Check cooldown
                if alert.last_triggered:
                    time_since_last = (DateTimeHelpers.now_utc() - alert.last_triggered).total_seconds()
                    if time_since_last < alert.cooldown_seconds:
                        continue
                
                # Get current metric value
                metric_values = self._metrics.get(alert.metric_name, deque())
                if not metric_values:
                    continue
                
                current_value = metric_values[-1].value
                
                # Check condition
                should_trigger = False
                if alert.condition == AlertCondition.GREATER_THAN:
                    should_trigger = current_value > alert.threshold
                elif alert.condition == AlertCondition.LESS_THAN:
                    should_trigger = current_value < alert.threshold
                elif alert.condition == AlertCondition.EQUAL_TO:
                    should_trigger = current_value == alert.threshold
                elif alert.condition == AlertCondition.NOT_EQUAL_TO:
                    should_trigger = current_value != alert.threshold
                elif alert.condition == AlertCondition.GREATER_THAN_OR_EQUAL:
                    should_trigger = current_value >= alert.threshold
                elif alert.condition == AlertCondition.LESS_THAN_OR_EQUAL:
                    should_trigger = current_value <= alert.threshold
                
                if should_trigger:
                    await self._trigger_alert(alert, current_value)
        
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
    
    async def _trigger_alert(self, alert: Alert, current_value: float):
        """Trigger an alert"""
        try:
            # Create alert event
            alert_event = AlertEvent(
                alert_id=alert.id,
                alert_name=alert.name,
                level=alert.level,
                message=f"{alert.description}. Current value: {current_value:.2f}, Threshold: {alert.threshold:.2f}",
                metric_value=current_value,
                threshold=alert.threshold
            )
            
            # Store alert event
            self._alert_events.append(alert_event)
            
            # Update alert
            alert.last_triggered = DateTimeHelpers.now_utc()
            alert.trigger_count += 1
            
            # Notify alert handlers
            for handler in self._alert_handlers:
                try:
                    handler(alert_event)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")
            
            logger.warning(f"Alert triggered: {alert.name} - {alert_event.message}")
        
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
    
    async def _update_system_health(self):
        """Update system health status"""
        try:
            # Get current metrics
            cpu_metrics = self._metrics.get("system.cpu.percent", deque())
            memory_metrics = self._metrics.get("system.memory.percent", deque())
            disk_metrics = self._metrics.get("system.disk.percent", deque())
            network_metrics = self._metrics.get("system.network.bytes_sent", deque())
            
            # Calculate averages
            cpu_usage = cpu_metrics[-1].value if cpu_metrics else 0
            memory_usage = memory_metrics[-1].value if memory_metrics else 0
            disk_usage = disk_metrics[-1].value if disk_metrics else 0
            
            # Calculate network I/O
            network_io = {}
            if network_metrics:
                network_io["bytes_sent"] = network_metrics[-1].value
                recv_metrics = self._metrics.get("system.network.bytes_recv", deque())
                if recv_metrics:
                    network_io["bytes_recv"] = recv_metrics[-1].value
            
            # Determine overall status
            status = "healthy"
            if cpu_usage > 90 or memory_usage > 95 or disk_usage > 95:
                status = "critical"
            elif cpu_usage > 80 or memory_usage > 85 or disk_usage > 90:
                status = "warning"
            elif cpu_usage > 70 or memory_usage > 75 or disk_usage > 80:
                status = "degraded"
            
            # Create health record
            health = SystemHealth(
                status=status,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=0,  # Would be calculated from actual connections
                response_time=0.0,  # Would be calculated from actual response times
                error_rate=0.0  # Would be calculated from actual error rates
            )
            
            self._health_history.append(health)
        
        except Exception as e:
            logger.error(f"Failed to update system health: {e}")
    
    def add_alert(self, alert: Alert):
        """Add a new alert"""
        self._alerts[alert.id] = alert
        logger.info(f"Alert added: {alert.name}")
    
    def remove_alert(self, alert_id: str):
        """Remove an alert"""
        if alert_id in self._alerts:
            del self._alerts[alert_id]
            logger.info(f"Alert removed: {alert_id}")
    
    def update_alert(self, alert_id: str, **kwargs):
        """Update an alert"""
        if alert_id in self._alerts:
            alert = self._alerts[alert_id]
            for key, value in kwargs.items():
                if hasattr(alert, key):
                    setattr(alert, key, value)
            logger.info(f"Alert updated: {alert_id}")
    
    def add_alert_handler(self, handler: Callable[[AlertEvent], None]):
        """Add alert handler"""
        self._alert_handlers.append(handler)
    
    def add_metric_handler(self, handler: Callable[[Metric], None]):
        """Add metric handler"""
        self._metric_handlers.append(handler)
    
    def get_metrics(self, metric_name: str, limit: int = 100) -> List[Metric]:
        """Get metrics for a specific metric name"""
        metrics = self._metrics.get(metric_name, deque())
        return list(metrics)[-limit:]
    
    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        """Get all metrics"""
        return {name: list(metrics) for name, metrics in self._metrics.items()}
    
    def get_alert_events(self, limit: int = 100) -> List[AlertEvent]:
        """Get alert events"""
        return self._alert_events[-limit:]
    
    def get_system_health_history(self, limit: int = 100) -> List[SystemHealth]:
        """Get system health history"""
        return list(self._health_history)[-limit:]
    
    def get_current_health(self) -> Optional[SystemHealth]:
        """Get current system health"""
        return self._health_history[-1] if self._health_history else None
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            summary = {}
            
            for metric_name, metrics in self._metrics.items():
                if not metrics:
                    continue
                
                values = [m.value for m in metrics]
                summary[metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": statistics.mean(values),
                    "median": statistics.median(values),
                    "latest": values[-1] if values else 0,
                    "timestamp": metrics[-1].timestamp.isoformat() if metrics else None
                }
            
            return {
                "metrics": summary,
                "total_metrics": len(self._metrics),
                "active_alerts": len([a for a in self._alerts.values() if a.is_active]),
                "total_alerts": len(self._alerts),
                "alert_events": len(self._alert_events),
                "timestamp": DateTimeHelpers.now_utc().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}
    
    async def get_health_dashboard(self) -> Dict[str, Any]:
        """Get health dashboard data"""
        try:
            current_health = self.get_current_health()
            recent_alerts = self.get_alert_events(10)
            health_history = self.get_system_health_history(24)  # Last 24 hours
            
            # Calculate uptime (simplified)
            uptime_hours = 24  # Would be calculated from actual start time
            
            # Calculate availability
            healthy_count = len([h for h in health_history if h.status == "healthy"])
            availability = (healthy_count / len(health_history)) * 100 if health_history else 100
            
            return {
                "current_health": {
                    "status": current_health.status if current_health else "unknown",
                    "cpu_usage": current_health.cpu_usage if current_health else 0,
                    "memory_usage": current_health.memory_usage if current_health else 0,
                    "disk_usage": current_health.disk_usage if current_health else 0
                },
                "recent_alerts": [
                    {
                        "alert_name": alert.alert_name,
                        "level": alert.level.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in recent_alerts
                ],
                "system_stats": {
                    "uptime_hours": uptime_hours,
                    "availability_percent": availability,
                    "total_alerts": len(self._alerts),
                    "active_alerts": len([a for a in self._alerts.values() if a.is_active]),
                    "total_metrics": len(self._metrics)
                },
                "timestamp": DateTimeHelpers.now_utc().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get health dashboard: {e}")
            return {"error": str(e)}


# Global monitoring service
monitoring_service = MonitoringService()


# Utility functions
async def start_monitoring_service():
    """Start the monitoring service"""
    await monitoring_service.start()


async def stop_monitoring_service():
    """Stop the monitoring service"""
    await monitoring_service.stop()


def add_alert(alert: Alert):
    """Add a new alert"""
    monitoring_service.add_alert(alert)


def remove_alert(alert_id: str):
    """Remove an alert"""
    monitoring_service.remove_alert(alert_id)


def update_alert(alert_id: str, **kwargs):
    """Update an alert"""
    monitoring_service.update_alert(alert_id, **kwargs)


def add_alert_handler(handler: Callable[[AlertEvent], None]):
    """Add alert handler"""
    monitoring_service.add_alert_handler(handler)


def add_metric_handler(handler: Callable[[Metric], None]):
    """Add metric handler"""
    monitoring_service.add_metric_handler(handler)


async def record_metric(name: str, value: float, metric_type: MetricType = MetricType.GAUGE, labels: Optional[Dict[str, str]] = None):
    """Record a metric"""
    await monitoring_service._record_metric(name, value, metric_type, labels)


def get_metrics(metric_name: str, limit: int = 100) -> List[Metric]:
    """Get metrics for a specific metric name"""
    return monitoring_service.get_metrics(metric_name, limit)


def get_all_metrics() -> Dict[str, List[Metric]]:
    """Get all metrics"""
    return monitoring_service.get_all_metrics()


def get_alert_events(limit: int = 100) -> List[AlertEvent]:
    """Get alert events"""
    return monitoring_service.get_alert_events(limit)


def get_system_health_history(limit: int = 100) -> List[SystemHealth]:
    """Get system health history"""
    return monitoring_service.get_system_health_history(limit)


def get_current_health() -> Optional[SystemHealth]:
    """Get current system health"""
    return monitoring_service.get_current_health()


async def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary"""
    return await monitoring_service.get_metrics_summary()


async def get_health_dashboard() -> Dict[str, Any]:
    """Get health dashboard data"""
    return await monitoring_service.get_health_dashboard()


# Monitoring decorators
def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            await record_metric(f"function.{func.__name__}.duration", duration, MetricType.TIMER)
    
    return wrapper


def monitor_errors(func: Callable) -> Callable:
    """Decorator to monitor function errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            await record_metric(f"function.{func.__name__}.success", 1, MetricType.COUNTER)
            return result
        except Exception as e:
            await record_metric(f"function.{func.__name__}.error", 1, MetricType.COUNTER)
            raise
    
    return wrapper


