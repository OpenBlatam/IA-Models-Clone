"""
Advanced Monitoring System
==========================

Advanced monitoring with alerts and dashboards.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert level."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert."""
    id: str
    level: AlertLevel
    message: str
    timestamp: float
    metadata: Dict[str, Any] = None
    resolved: bool = False
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Metric:
    """Metric."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class AdvancedMonitor:
    """Advanced monitoring system."""
    
    def __init__(self):
        """Initialize advanced monitor."""
        self.metrics: List[Metric] = []
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        self.metric_collectors: List[Callable] = []
    
    def register_alert_handler(self, handler: Callable) -> None:
        """
        Register alert handler.
        
        Args:
            handler: Alert handler function
        """
        self.alert_handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__name__}")
    
    def register_metric_collector(self, collector: Callable) -> None:
        """
        Register metric collector.
        
        Args:
            collector: Metric collector function
        """
        self.metric_collectors.append(collector)
        logger.info(f"Registered metric collector: {collector.__name__}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
        )
        
        self.metrics.append(metric)
        
        # Keep only last 10000 metrics
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-10000:]
        
        logger.debug(f"Recorded metric: {name} = {value}")
    
    def create_alert(
        self,
        level: AlertLevel,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """
        Create an alert.
        
        Args:
            level: Alert level
            message: Alert message
            metadata: Optional metadata
            
        Returns:
            Created alert
        """
        alert = Alert(
            id=f"alert_{int(time.time() * 1000)}",
            level=level,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        logger.warning(f"Alert created: {level.value} - {message}")
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if resolved
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                return True
        
        return False
    
    def collect_metrics(self) -> None:
        """Collect metrics from registered collectors."""
        for collector in self.metric_collectors:
            try:
                metrics = collector()
                if isinstance(metrics, list):
                    for metric in metrics:
                        self.record_metric(
                            metric.get("name"),
                            metric.get("value"),
                            metric.get("tags"),
                        )
                elif isinstance(metrics, dict):
                    for name, value in metrics.items():
                        self.record_metric(name, value)
            except Exception as e:
                logger.error(f"Metric collector error: {e}")
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        time_range: Optional[float] = None,
    ) -> List[Metric]:
        """
        Get metrics.
        
        Args:
            name: Optional metric name filter
            time_range: Optional time range in seconds
            
        Returns:
            List of metrics
        """
        metrics = self.metrics
        
        if name:
            metrics = [m for m in metrics if m.name == name]
        
        if time_range:
            cutoff_time = time.time() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return metrics
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        resolved: Optional[bool] = None,
    ) -> List[Alert]:
        """
        Get alerts.
        
        Args:
            level: Optional alert level filter
            resolved: Optional resolved filter
            
        Returns:
            List of alerts
        """
        alerts = self.alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return alerts
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        active_alerts = [a for a in self.alerts if not a.resolved]
        
        # Get recent metrics
        recent_metrics = self.metrics[-100:] if self.metrics else []
        
        # Group metrics by name
        metrics_by_name = {}
        for metric in recent_metrics:
            if metric.name not in metrics_by_name:
                metrics_by_name[metric.name] = []
            metrics_by_name[metric.name].append(metric.value)
        
        # Calculate statistics
        metrics_stats = {}
        for name, values in metrics_by_name.items():
            metrics_stats[name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }
        
        return {
            "alerts": {
                "total": len(self.alerts),
                "active": len(active_alerts),
                "resolved": len([a for a in self.alerts if a.resolved]),
                "by_level": {
                    level.value: len([a for a in active_alerts if a.level == level])
                    for level in AlertLevel
                },
            },
            "metrics": {
                "total": len(self.metrics),
                "recent": len(recent_metrics),
                "statistics": metrics_stats,
            },
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "metrics_count": len(self.metrics),
            "alerts_count": len(self.alerts),
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "alert_handlers": len(self.alert_handlers),
            "metric_collectors": len(self.metric_collectors),
        }

