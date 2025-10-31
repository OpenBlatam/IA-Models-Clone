"""
Refactored Metrics System

Sistema de métricas y monitoreo refactorizado para el AI History Comparison System.
Maneja métricas en tiempo real, alertas inteligentes, dashboards dinámicos y optimización automática.
"""

import asyncio
import logging
import time
import statistics
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from contextlib import asynccontextmanager
import json
import weakref
from collections import deque, defaultdict
import math

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"
    RATE = "rate"
    PERCENTILE = "percentile"
    CUSTOM = "custom"


class MetricUnit(Enum):
    """Metric unit enumeration"""
    COUNT = "count"
    BYTES = "bytes"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    PERCENT = "percent"
    RATIO = "ratio"
    CUSTOM = "custom"


class AlertSeverity(Enum):
    """Alert severity enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricAggregation(Enum):
    """Metric aggregation enumeration"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    RATE = "rate"


@dataclass
class MetricMetadata:
    """Metric metadata"""
    name: str
    metric_type: MetricType
    unit: MetricUnit
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    aggregation: MetricAggregation = MetricAggregation.AVG
    retention_period: timedelta = timedelta(days=7)
    sampling_rate: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetricValue:
    """Metric value with timestamp"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Optional[MetricMetadata] = None


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # e.g., "value > 100", "rate > 0.5"
    severity: AlertSeverity
    duration: timedelta = timedelta(minutes=5)
    cooldown: timedelta = timedelta(minutes=15)
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Alert:
    """Alert instance"""
    rule: AlertRule
    metric_value: MetricValue
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    is_active: bool = True
    notification_sent: bool = False


class MetricCollector(ABC):
    """Abstract metric collector"""
    
    @abstractmethod
    async def collect(self) -> List[MetricValue]:
        """Collect metrics"""
        pass


class SystemMetricCollector(MetricCollector):
    """System metrics collector"""
    
    async def collect(self) -> List[MetricValue]:
        """Collect system metrics"""
        import psutil
        
        metrics = []
        current_time = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(MetricValue(
            value=cpu_percent,
            timestamp=current_time,
            labels={"type": "cpu_percent"},
            metadata=MetricMetadata(
                name="system.cpu.percent",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENT
            )
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(MetricValue(
            value=memory.percent,
            timestamp=current_time,
            labels={"type": "memory_percent"},
            metadata=MetricMetadata(
                name="system.memory.percent",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENT
            )
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(MetricValue(
            value=disk.percent,
            timestamp=current_time,
            labels={"type": "disk_percent"},
            metadata=MetricMetadata(
                name="system.disk.percent",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENT
            )
        ))
        
        return metrics


class ApplicationMetricCollector(MetricCollector):
    """Application metrics collector"""
    
    def __init__(self):
        self._request_count = 0
        self._error_count = 0
        self._response_times = deque(maxlen=1000)
    
    async def collect(self) -> List[MetricValue]:
        """Collect application metrics"""
        metrics = []
        current_time = datetime.utcnow()
        
        # Request count
        metrics.append(MetricValue(
            value=self._request_count,
            timestamp=current_time,
            labels={"type": "request_count"},
            metadata=MetricMetadata(
                name="application.requests.total",
                metric_type=MetricType.COUNTER,
                unit=MetricUnit.COUNT
            )
        ))
        
        # Error count
        metrics.append(MetricValue(
            value=self._error_count,
            timestamp=current_time,
            labels={"type": "error_count"},
            metadata=MetricMetadata(
                name="application.errors.total",
                metric_type=MetricType.COUNTER,
                unit=MetricUnit.COUNT
            )
        ))
        
        # Response time
        if self._response_times:
            avg_response_time = statistics.mean(self._response_times)
            metrics.append(MetricValue(
                value=avg_response_time,
                timestamp=current_time,
                labels={"type": "response_time"},
                metadata=MetricMetadata(
                    name="application.response_time.avg",
                    metric_type=MetricType.GAUGE,
                    unit=MetricUnit.MILLISECONDS
                )
            ))
        
        return metrics
    
    def record_request(self, response_time: float, is_error: bool = False):
        """Record request metric"""
        self._request_count += 1
        if is_error:
            self._error_count += 1
        self._response_times.append(response_time)


class MetricStorage:
    """Metric storage with retention and aggregation"""
    
    def __init__(self, max_samples: int = 10000):
        self._storage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._metadata: Dict[str, MetricMetadata] = {}
        self._lock = asyncio.Lock()
    
    async def store(self, metric: MetricValue) -> None:
        """Store metric value"""
        async with self._lock:
            key = self._get_storage_key(metric)
            self._storage[key].append(metric)
            
            if metric.metadata:
                self._metadata[key] = metric.metadata
    
    def _get_storage_key(self, metric: MetricValue) -> str:
        """Get storage key for metric"""
        if metric.metadata:
            return metric.metadata.name
        return "unknown"
    
    async def get_metrics(self, name: str, start_time: datetime = None, 
                         end_time: datetime = None) -> List[MetricValue]:
        """Get metrics by name and time range"""
        async with self._lock:
            if name not in self._storage:
                return []
            
            metrics = list(self._storage[name])
            
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            return metrics
    
    async def get_aggregated_metrics(self, name: str, aggregation: MetricAggregation,
                                   start_time: datetime = None, end_time: datetime = None) -> float:
        """Get aggregated metric value"""
        metrics = await self.get_metrics(name, start_time, end_time)
        if not metrics:
            return 0.0
        
        values = [m.value for m in metrics]
        
        if aggregation == MetricAggregation.SUM:
            return sum(values)
        elif aggregation == MetricAggregation.AVG:
            return statistics.mean(values)
        elif aggregation == MetricAggregation.MIN:
            return min(values)
        elif aggregation == MetricAggregation.MAX:
            return max(values)
        elif aggregation == MetricAggregation.COUNT:
            return len(values)
        elif aggregation == MetricAggregation.MEDIAN:
            return statistics.median(values)
        else:
            return statistics.mean(values)
    
    async def cleanup_expired_metrics(self) -> None:
        """Cleanup expired metrics"""
        async with self._lock:
            current_time = datetime.utcnow()
            
            for name, metrics in self._storage.items():
                if name in self._metadata:
                    retention_period = self._metadata[name].retention_period
                    cutoff_time = current_time - retention_period
                    
                    # Remove expired metrics
                    while metrics and metrics[0].timestamp < cutoff_time:
                        metrics.popleft()


class AlertManager:
    """Alert manager with rule evaluation and notification"""
    
    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._notification_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
    
    async def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule"""
        async with self._lock:
            self._rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    async def remove_rule(self, name: str) -> None:
        """Remove alert rule"""
        async with self._lock:
            if name in self._rules:
                del self._rules[name]
                logger.info(f"Removed alert rule: {name}")
    
    async def evaluate_metrics(self, metrics: List[MetricValue]) -> None:
        """Evaluate metrics against alert rules"""
        async with self._lock:
            for metric in metrics:
                for rule_name, rule in self._rules.items():
                    if not rule.enabled or rule.metric_name != metric.metadata.name:
                        continue
                    
                    await self._evaluate_rule(rule, metric)
    
    async def _evaluate_rule(self, rule: AlertRule, metric: MetricValue) -> None:
        """Evaluate single rule against metric"""
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            condition_met = self._evaluate_condition(rule.condition, metric.value)
            
            if condition_met:
                await self._trigger_alert(rule, metric)
            else:
                await self._resolve_alert(rule.name)
        
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _evaluate_condition(self, condition: str, value: float) -> bool:
        """Evaluate condition string"""
        # Simple condition evaluation (in production, use a proper expression evaluator)
        if ">" in condition:
            parts = condition.split(">")
            if len(parts) == 2:
                threshold = float(parts[1].strip())
                return value > threshold
        elif "<" in condition:
            parts = condition.split("<")
            if len(parts) == 2:
                threshold = float(parts[1].strip())
                return value < threshold
        elif "==" in condition:
            parts = condition.split("==")
            if len(parts) == 2:
                threshold = float(parts[1].strip())
                return value == threshold
        
        return False
    
    async def _trigger_alert(self, rule: AlertRule, metric: MetricValue) -> None:
        """Trigger alert"""
        alert_key = f"{rule.name}_{metric.metadata.name}"
        
        if alert_key in self._active_alerts:
            return  # Alert already active
        
        alert = Alert(
            rule=rule,
            metric_value=metric,
            triggered_at=datetime.utcnow()
        )
        
        self._active_alerts[alert_key] = alert
        
        # Send notification
        await self._send_notification(alert)
        
        logger.warning(f"Alert triggered: {rule.name} - {metric.value}")
    
    async def _resolve_alert(self, rule_name: str) -> None:
        """Resolve alert"""
        alert_key = None
        for key, alert in self._active_alerts.items():
            if alert.rule.name == rule_name:
                alert_key = key
                break
        
        if alert_key:
            alert = self._active_alerts[alert_key]
            alert.resolved_at = datetime.utcnow()
            alert.is_active = False
            
            logger.info(f"Alert resolved: {rule_name}")
    
    async def _send_notification(self, alert: Alert) -> None:
        """Send alert notification"""
        for handler in self._notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
    
    def add_notification_handler(self, handler: Callable) -> None:
        """Add notification handler"""
        self._notification_handlers.append(handler)
    
    def remove_notification_handler(self, handler: Callable) -> None:
        """Remove notification handler"""
        if handler in self._notification_handlers:
            self._notification_handlers.remove(handler)


class RefactoredMetricsManager:
    """Refactored metrics manager with advanced features"""
    
    def __init__(self):
        self._collectors: List[MetricCollector] = []
        self._storage = MetricStorage()
        self._alert_manager = AlertManager()
        self._custom_metrics: Dict[str, MetricValue] = {}
        self._collection_interval: float = 30.0  # 30 seconds
        self._collection_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval: float = 300.0  # 5 minutes
        self._lock = asyncio.Lock()
        self._callbacks: List[Callable] = []
    
    async def initialize(self) -> None:
        """Initialize metrics manager"""
        # Add default collectors
        self._collectors.append(SystemMetricCollector())
        self._collectors.append(ApplicationMetricCollector())
        
        # Start collection task
        self._collection_task = asyncio.create_task(self._collection_loop())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Refactored metrics manager initialized")
    
    async def _collection_loop(self) -> None:
        """Metric collection loop"""
        while True:
            try:
                await asyncio.sleep(self._collection_interval)
                await self._collect_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._storage.cleanup_expired_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _collect_metrics(self) -> None:
        """Collect metrics from all collectors"""
        all_metrics = []
        
        for collector in self._collectors:
            try:
                metrics = await collector.collect()
                all_metrics.extend(metrics)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
        
        # Store metrics
        for metric in all_metrics:
            await self._storage.store(metric)
        
        # Evaluate alerts
        await self._alert_manager.evaluate_metrics(all_metrics)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(all_metrics)
                else:
                    callback(all_metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    async def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                          unit: MetricUnit = MetricUnit.COUNT, labels: Dict[str, str] = None) -> None:
        """Record custom metric"""
        metric = MetricValue(
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            metadata=MetricMetadata(
                name=name,
                metric_type=metric_type,
                unit=unit
            )
        )
        
        await self._storage.store(metric)
        self._custom_metrics[name] = metric
    
    async def get_metric(self, name: str, aggregation: MetricAggregation = MetricAggregation.AVG,
                        start_time: datetime = None, end_time: datetime = None) -> float:
        """Get metric value"""
        return await self._storage.get_aggregated_metrics(name, aggregation, start_time, end_time)
    
    async def get_metrics_history(self, name: str, start_time: datetime = None,
                                 end_time: datetime = None) -> List[MetricValue]:
        """Get metrics history"""
        return await self._storage.get_metrics(name, start_time, end_time)
    
    async def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule"""
        await self._alert_manager.add_rule(rule)
    
    async def remove_alert_rule(self, name: str) -> None:
        """Remove alert rule"""
        await self._alert_manager.remove_rule(name)
    
    def add_notification_handler(self, handler: Callable) -> None:
        """Add notification handler"""
        self._alert_manager.add_notification_handler(handler)
    
    def add_callback(self, callback: Callable) -> None:
        """Add metrics callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove metrics callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        current_time = datetime.utcnow()
        one_hour_ago = current_time - timedelta(hours=1)
        
        dashboard_data = {
            "timestamp": current_time.isoformat(),
            "metrics": {},
            "alerts": {
                "active": len([a for a in self._alert_manager._active_alerts.values() if a.is_active]),
                "total": len(self._alert_manager._active_alerts)
            },
            "system": {
                "collectors": len(self._collectors),
                "custom_metrics": len(self._custom_metrics)
            }
        }
        
        # Get key metrics
        key_metrics = [
            "system.cpu.percent",
            "system.memory.percent",
            "system.disk.percent",
            "application.requests.total",
            "application.errors.total",
            "application.response_time.avg"
        ]
        
        for metric_name in key_metrics:
            try:
                current_value = await self.get_metric(metric_name, MetricAggregation.AVG)
                history = await self.get_metrics_history(metric_name, one_hour_ago, current_time)
                
                dashboard_data["metrics"][metric_name] = {
                    "current": current_value,
                    "history": [
                        {"timestamp": m.timestamp.isoformat(), "value": m.value}
                        for m in history[-20:]  # Last 20 points
                    ]
                }
            except Exception as e:
                logger.error(f"Error getting metric {metric_name}: {e}")
        
        return dashboard_data
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get metrics manager health status"""
        return {
            "collectors_count": len(self._collectors),
            "custom_metrics_count": len(self._custom_metrics),
            "collection_interval": self._collection_interval,
            "cleanup_interval": self._cleanup_interval,
            "active_alerts": len([a for a in self._alert_manager._active_alerts.values() if a.is_active]),
            "total_alerts": len(self._alert_manager._active_alerts),
            "alert_rules": len(self._alert_manager._rules)
        }
    
    async def shutdown(self) -> None:
        """Shutdown metrics manager"""
        if self._collection_task:
            self._collection_task.cancel()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info("Refactored metrics manager shutdown")


# Global metrics manager
metrics_manager = RefactoredMetricsManager()


# Convenience functions
async def record_metric(name: str, value: float, **kwargs):
    """Record metric"""
    await metrics_manager.record_metric(name, value, **kwargs)


async def get_metric(name: str, **kwargs):
    """Get metric value"""
    return await metrics_manager.get_metric(name, **kwargs)


async def add_alert_rule(rule: AlertRule):
    """Add alert rule"""
    await metrics_manager.add_alert_rule(rule)


# Metric decorators
def metric(name: str, metric_type: MetricType = MetricType.GAUGE, unit: MetricUnit = MetricUnit.COUNT):
    """Metric decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                await record_metric(name, 1, metric_type, unit)
                return result
            except Exception as e:
                await record_metric(f"{name}.error", 1, MetricType.COUNTER, MetricUnit.COUNT)
                raise
            finally:
                duration = (time.time() - start_time) * 1000  # milliseconds
                await record_metric(f"{name}.duration", duration, MetricType.TIMER, MetricUnit.MILLISECONDS)
        return wrapper
    return decorator


def timer(name: str):
    """Timer decorator"""
    return metric(name, MetricType.TIMER, MetricUnit.MILLISECONDS)


def counter(name: str):
    """Counter decorator"""
    return metric(name, MetricType.COUNTER, MetricUnit.COUNT)


def gauge(name: str):
    """Gauge decorator"""
    return metric(name, MetricType.GAUGE, MetricUnit.COUNT)





















