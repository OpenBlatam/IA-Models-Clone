"""
Advanced Monitoring System for Facebook Posts
Following functional programming principles and monitoring best practices
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)


# Pure functions for monitoring

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass(frozen=True)
class MetricData:
    """Immutable metric data - pure data structure"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass(frozen=True)
class LogEntry:
    """Immutable log entry - pure data structure"""
    level: LogLevel
    message: str
    module: str
    function: str
    timestamp: datetime
    extra_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "level": self.level.value,
            "message": self.message,
            "module": self.module,
            "function": self.function,
            "timestamp": self.timestamp.isoformat(),
            "extra_data": self.extra_data
        }


@dataclass(frozen=True)
class AlertRule:
    """Immutable alert rule - pure data structure"""
    name: str
    metric_name: str
    condition: str
    threshold: Union[int, float]
    severity: str
    enabled: bool
    
    def evaluate(self, metric_value: Union[int, float]) -> bool:
        """Evaluate alert rule - pure function"""
        if not self.enabled:
            return False
        
        if self.condition == "greater_than":
            return metric_value > self.threshold
        elif self.condition == "less_than":
            return metric_value < self.threshold
        elif self.condition == "equals":
            return metric_value == self.threshold
        elif self.condition == "not_equals":
            return metric_value != self.threshold
        else:
            return False


def create_metric_data(
    name: str,
    value: Union[int, float],
    metric_type: MetricType,
    labels: Optional[Dict[str, str]] = None
) -> MetricData:
    """Create metric data - pure function"""
    return MetricData(
        name=name,
        value=value,
        metric_type=metric_type,
        labels=labels or {},
        timestamp=datetime.utcnow()
    )


def create_log_entry(
    level: LogLevel,
    message: str,
    module: str,
    function: str,
    extra_data: Optional[Dict[str, Any]] = None
) -> LogEntry:
    """Create log entry - pure function"""
    return LogEntry(
        level=level,
        message=message,
        module=module,
        function=function,
        timestamp=datetime.utcnow(),
        extra_data=extra_data or {}
    )


def create_alert_rule(
    name: str,
    metric_name: str,
    condition: str,
    threshold: Union[int, float],
    severity: str = "warning"
) -> AlertRule:
    """Create alert rule - pure function"""
    return AlertRule(
        name=name,
        metric_name=metric_name,
        condition=condition,
        threshold=threshold,
        severity=severity,
        enabled=True
    )


def calculate_metric_statistics(metrics: List[MetricData]) -> Dict[str, Any]:
    """Calculate metric statistics - pure function"""
    if not metrics:
        return {"count": 0, "average": 0, "min": 0, "max": 0, "sum": 0}
    
    values = [m.value for m in metrics if isinstance(m.value, (int, float))]
    
    if not values:
        return {"count": 0, "average": 0, "min": 0, "max": 0, "sum": 0}
    
    return {
        "count": len(values),
        "average": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
        "last_value": values[-1] if values else 0
    }


def filter_metrics_by_time_range(
    metrics: List[MetricData],
    start_time: datetime,
    end_time: datetime
) -> List[MetricData]:
    """Filter metrics by time range - pure function"""
    return [
        metric for metric in metrics
        if start_time <= metric.timestamp <= end_time
    ]


def group_metrics_by_name(metrics: List[MetricData]) -> Dict[str, List[MetricData]]:
    """Group metrics by name - pure function"""
    grouped = defaultdict(list)
    for metric in metrics:
        grouped[metric.name].append(metric)
    return dict(grouped)


def calculate_metric_trends(metrics: List[MetricData]) -> Dict[str, str]:
    """Calculate metric trends - pure function"""
    if len(metrics) < 2:
        return {"trend": "insufficient_data"}
    
    values = [m.value for m in metrics if isinstance(m.value, (int, float))]
    
    if len(values) < 2:
        return {"trend": "insufficient_data"}
    
    # Simple trend calculation
    first_half = values[:len(values)//2]
    second_half = values[len(values)//2:]
    
    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)
    
    if second_avg > first_avg * 1.1:
        trend = "increasing"
    elif second_avg < first_avg * 0.9:
        trend = "decreasing"
    else:
        trend = "stable"
    
    return {
        "trend": trend,
        "first_half_average": first_avg,
        "second_half_average": second_avg,
        "change_percentage": ((second_avg - first_avg) / first_avg) * 100
    }


# Advanced Monitoring System Class

class AdvancedMonitoringSystem:
    """Advanced Monitoring System following functional principles"""
    
    def __init__(self, max_metrics: int = 10000, max_logs: int = 5000):
        self.max_metrics = max_metrics
        self.max_logs = max_logs
        
        # Storage
        self.metrics: deque = deque(maxlen=max_metrics)
        self.logs: deque = deque(maxlen=max_logs)
        self.alerts: List[AlertRule] = []
        self.active_alerts: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            "total_metrics": 0,
            "total_logs": 0,
            "total_alerts": 0,
            "active_alerts": 0
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.metric_callbacks: List[Callable] = []
    
    async def start_monitoring(self) -> None:
        """Start monitoring system"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Advanced monitoring system started")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring system"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced monitoring system stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Check for alerts
                await self._check_alerts()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Wait for next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(5)
    
    async def _check_alerts(self) -> None:
        """Check for alert conditions"""
        for alert_rule in self.alerts:
            if not alert_rule.enabled:
                continue
            
            # Get recent metrics for this alert
            recent_metrics = [
                m for m in self.metrics
                if m.name == alert_rule.metric_name
                and m.timestamp > datetime.utcnow() - timedelta(minutes=5)
            ]
            
            if not recent_metrics:
                continue
            
            # Check if alert condition is met
            latest_metric = recent_metrics[-1]
            if alert_rule.evaluate(latest_metric.value):
                await self._trigger_alert(alert_rule, latest_metric)
    
    async def _trigger_alert(self, alert_rule: AlertRule, metric: MetricData) -> None:
        """Trigger alert"""
        alert_data = {
            "rule_name": alert_rule.name,
            "metric_name": alert_rule.metric_name,
            "metric_value": metric.value,
            "threshold": alert_rule.threshold,
            "condition": alert_rule.condition,
            "severity": alert_rule.severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add to active alerts
        self.active_alerts.append(alert_data)
        self.stats["active_alerts"] = len(self.active_alerts)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.error("Error in alert callback", error=str(e))
        
        logger.warning(f"Alert triggered: {alert_rule.name}", alert_data=alert_data)
    
    async def _cleanup_old_data(self) -> None:
        """Cleanup old data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Cleanup old metrics
        old_metrics_count = len(self.metrics)
        while self.metrics and self.metrics[0].timestamp < cutoff_time:
            self.metrics.popleft()
        
        # Cleanup old logs
        old_logs_count = len(self.logs)
        while self.logs and self.logs[0].timestamp < cutoff_time:
            self.logs.popleft()
        
        # Cleanup old alerts
        self.active_alerts = [
            alert for alert in self.active_alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]
        
        self.stats["active_alerts"] = len(self.active_alerts)
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric"""
        metric = create_metric_data(name, value, metric_type, labels)
        self.metrics.append(metric)
        self.stats["total_metrics"] += 1
        
        # Call metric callbacks
        for callback in self.metric_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(metric))
                else:
                    callback(metric)
            except Exception as e:
                logger.error("Error in metric callback", error=str(e))
    
    def record_log(
        self,
        level: LogLevel,
        message: str,
        module: str,
        function: str,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a log entry"""
        log_entry = create_log_entry(level, message, module, function, extra_data)
        self.logs.append(log_entry)
        self.stats["total_logs"] += 1
    
    def add_alert_rule(self, alert_rule: AlertRule) -> None:
        """Add alert rule"""
        self.alerts.append(alert_rule)
        self.stats["total_alerts"] += 1
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove alert rule"""
        for i, rule in enumerate(self.alerts):
            if rule.name == rule_name:
                del self.alerts[i]
                return True
        return False
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def add_metric_callback(self, callback: Callable) -> None:
        """Add metric callback"""
        self.metric_callbacks.append(callback)
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get metrics with optional filtering"""
        metrics = list(self.metrics)
        
        # Filter by name
        if name:
            metrics = [m for m in metrics if m.name == name]
        
        # Filter by time range
        if start_time and end_time:
            metrics = filter_metrics_by_time_range(metrics, start_time, end_time)
        
        # Limit results
        metrics = metrics[-limit:] if limit else metrics
        
        return [metric.to_dict() for metric in metrics]
    
    def get_logs(
        self,
        level: Optional[LogLevel] = None,
        module: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get logs with optional filtering"""
        logs = list(self.logs)
        
        # Filter by level
        if level:
            logs = [l for l in logs if l.level == level]
        
        # Filter by module
        if module:
            logs = [l for l in logs if l.module == module]
        
        # Filter by time range
        if start_time and end_time:
            logs = [l for l in logs if start_time <= l.timestamp <= end_time]
        
        # Limit results
        logs = logs[-limit:] if limit else logs
        
        return [log.to_dict() for log in logs]
    
    def get_metric_statistics(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get metric statistics"""
        metrics = [
            m for m in self.metrics
            if m.name == name
        ]
        
        if start_time and end_time:
            metrics = filter_metrics_by_time_range(metrics, start_time, end_time)
        
        return calculate_metric_statistics(metrics)
    
    def get_metric_trends(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get metric trends"""
        metrics = [
            m for m in self.metrics
            if m.name == name
        ]
        
        if start_time and end_time:
            metrics = filter_metrics_by_time_range(metrics, start_time, end_time)
        
        return calculate_metric_trends(metrics)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        return {
            "statistics": self.stats.copy(),
            "active_alerts": self.active_alerts,
            "recent_metrics": self.get_metrics(limit=100),
            "recent_logs": self.get_logs(limit=100),
            "alert_rules": [asdict(rule) for rule in self.alerts],
            "monitoring_status": "active" if self.is_monitoring else "inactive",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        recent_errors = [
            log for log in self.logs
            if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]
            and log.timestamp > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        return {
            "status": "healthy" if len(recent_errors) == 0 else "degraded",
            "error_count": len(recent_errors),
            "active_alerts": len(self.active_alerts),
            "monitoring_active": self.is_monitoring,
            "total_metrics": self.stats["total_metrics"],
            "total_logs": self.stats["total_logs"],
            "timestamp": datetime.utcnow().isoformat()
        }


# Factory functions

def create_monitoring_system(max_metrics: int = 10000, max_logs: int = 5000) -> AdvancedMonitoringSystem:
    """Create monitoring system instance - pure function"""
    return AdvancedMonitoringSystem(max_metrics, max_logs)


async def get_monitoring_system() -> AdvancedMonitoringSystem:
    """Get monitoring system instance with monitoring started"""
    system = create_monitoring_system()
    await system.start_monitoring()
    return system


# Monitoring decorators

def monitor_function(monitoring_system: AdvancedMonitoringSystem):
    """Decorator to monitor function execution"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            module_name = func.__module__
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record success metrics
                monitoring_system.record_metric(
                    f"function.{function_name}.execution_time",
                    execution_time,
                    MetricType.HISTOGRAM
                )
                monitoring_system.record_metric(
                    f"function.{function_name}.success_count",
                    1,
                    MetricType.COUNTER
                )
                
                # Record log
                monitoring_system.record_log(
                    LogLevel.INFO,
                    f"Function {function_name} executed successfully",
                    module_name,
                    function_name,
                    {"execution_time": execution_time}
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error metrics
                monitoring_system.record_metric(
                    f"function.{function_name}.error_count",
                    1,
                    MetricType.COUNTER
                )
                monitoring_system.record_metric(
                    f"function.{function_name}.execution_time",
                    execution_time,
                    MetricType.HISTOGRAM
                )
                
                # Record error log
                monitoring_system.record_log(
                    LogLevel.ERROR,
                    f"Function {function_name} failed: {str(e)}",
                    module_name,
                    function_name,
                    {"error": str(e), "execution_time": execution_time}
                )
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            module_name = func.__module__
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record success metrics
                monitoring_system.record_metric(
                    f"function.{function_name}.execution_time",
                    execution_time,
                    MetricType.HISTOGRAM
                )
                monitoring_system.record_metric(
                    f"function.{function_name}.success_count",
                    1,
                    MetricType.COUNTER
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error metrics
                monitoring_system.record_metric(
                    f"function.{function_name}.error_count",
                    1,
                    MetricType.COUNTER
                )
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def monitor_api_endpoint(monitoring_system: AdvancedMonitoringSystem):
    """Decorator to monitor API endpoints"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint_name = func.__name__
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record API metrics
                monitoring_system.record_metric(
                    f"api.{endpoint_name}.response_time",
                    execution_time,
                    MetricType.HISTOGRAM
                )
                monitoring_system.record_metric(
                    f"api.{endpoint_name}.success_count",
                    1,
                    MetricType.COUNTER
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error metrics
                monitoring_system.record_metric(
                    f"api.{endpoint_name}.error_count",
                    1,
                    MetricType.COUNTER
                )
                monitoring_system.record_metric(
                    f"api.{endpoint_name}.response_time",
                    execution_time,
                    MetricType.HISTOGRAM
                )
                
                raise
        
        return wrapper
    
    return decorator

