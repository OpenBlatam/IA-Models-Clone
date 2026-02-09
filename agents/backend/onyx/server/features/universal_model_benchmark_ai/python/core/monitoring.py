"""
Monitoring Module - Enhanced real-time monitoring and alerting.

Provides:
- Real-time metrics collection with time windows
- Advanced alerting system with multiple conditions
- Comprehensive health checks
- Performance monitoring
- Alert handlers and notifications
- Metric aggregation and analysis
"""

import logging
import time
import statistics
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import threading

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCondition(str, Enum):
    """Alert condition types."""
    THRESHOLD = "threshold"
    RATE_OF_CHANGE = "rate_of_change"
    ANOMALY = "anomaly"
    TREND = "trend"


@dataclass
class Alert:
    """Enhanced alert definition."""
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: Optional[float] = None
    condition: AlertCondition = AlertCondition.THRESHOLD
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False
    resolved_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "condition": self.condition.value,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
            "metadata": self.metadata,
        }


@dataclass
class MetricSnapshot:
    """Metric snapshot for monitoring."""
    timestamp: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


class MetricCollector:
    """
    Enhanced real-time metric collector with statistical analysis.
    
    Features:
    - Rolling window metrics
    - Statistical analysis
    - Time-based queries
    - Metric aggregation
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metric collector.
        
        Args:
            window_size: Size of rolling window
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.lock = threading.Lock()
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for metric
        """
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.window_size)
            self.metrics[name].append((time.time(), value, labels or {}))
    
    def get_metric_stats(
        self,
        name: str,
        time_window: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Get comprehensive statistics for a metric.
        
        Args:
            name: Metric name
            time_window: Optional time window in seconds
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            
            # Filter by time window if specified
            now = time.time()
            data = self.metrics[name]
            
            if time_window:
                cutoff = now - time_window
                values = [v for t, v, _ in data if t >= cutoff]
            else:
                values = [v for _, v, _ in data]
            
            if not values:
                return None
            
            return {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "latest": values[-1] if values else 0.0,
                "p25": statistics.quantiles(values, n=4)[0] if len(values) > 3 else values[0],
                "p75": statistics.quantiles(values, n=4)[2] if len(values) > 3 else values[-1],
                "p95": statistics.quantiles(values, n=20)[18] if len(values) > 19 else values[-1],
            }
    
    def get_metric_timeseries(
        self,
        name: str,
        time_window: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """
        Get metric time series data.
        
        Args:
            name: Metric name
            time_window: Optional time window in seconds
        
        Returns:
            List of (timestamp, value) tuples
        """
        with self.lock:
            if name not in self.metrics:
                return []
            
            data = self.metrics[name]
            
            if time_window:
                now = time.time()
                cutoff = now - time_window
                return [(t, v) for t, v, _ in data if t >= cutoff]
            else:
                return [(t, v) for t, v, _ in data]
    
    def get_all_metrics(self) -> Dict[str, List[float]]:
        """Get all metrics as lists of values."""
        with self.lock:
            return {
                name: [v for _, v, _ in data]
                for name, data in self.metrics.items()
            }
    
    def get_metric_names(self) -> List[str]:
        """Get list of all metric names."""
        with self.lock:
            return list(self.metrics.keys())
    
    def clear_metric(self, name: str) -> None:
        """Clear a specific metric."""
        with self.lock:
            if name in self.metrics:
                del self.metrics[name]
    
    def clear_all(self) -> None:
        """Clear all metrics."""
        with self.lock:
            self.metrics.clear()


class AlertManager:
    """
    Enhanced alert management system.
    
    Features:
    - Multiple alert handlers
    - Alert resolution tracking
    - Rate of change detection
    - Anomaly detection
    - Alert deduplication
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: List[Alert] = []
        self.handlers: Dict[AlertLevel, List[Callable[[Alert], None]]] = {
            level: [] for level in AlertLevel
        }
        self.active_alerts: Dict[str, Alert] = {}  # For deduplication
        self.lock = threading.Lock()
        self.max_alerts = 10000  # Maximum number of alerts to keep
    
    def register_handler(
        self,
        level: AlertLevel,
        handler: Callable[[Alert], None],
    ) -> None:
        """
        Register alert handler.
        
        Args:
            level: Alert level
            handler: Handler function
        """
        with self.lock:
            self.handlers[level].append(handler)
    
    def trigger_alert(self, alert: Alert) -> None:
        """
        Trigger an alert.
        
        Args:
            alert: Alert to trigger
        """
        with self.lock:
            # Check for duplicate active alerts
            alert_key = f"{alert.metric}:{alert.level.value}"
            
            if alert_key in self.active_alerts:
                # Update existing alert
                existing = self.active_alerts[alert_key]
                existing.message = alert.message
                existing.value = alert.value
                existing.timestamp = alert.timestamp
                existing.metadata.update(alert.metadata)
                return
            
            # Add new alert
            self.alerts.append(alert)
            self.active_alerts[alert_key] = alert
            
            # Limit alert history
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
            
            # Call handlers
            for handler in self.handlers.get(alert.level, []):
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}", exc_info=True)
    
    def resolve_alert(
        self,
        metric: str,
        level: AlertLevel,
    ) -> Optional[Alert]:
        """
        Resolve an active alert.
        
        Args:
            metric: Metric name
            level: Alert level
        
        Returns:
            Resolved alert if found
        """
        with self.lock:
            alert_key = f"{metric}:{level.value}"
            
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.resolved = True
                alert.resolved_at = datetime.now().isoformat()
                del self.active_alerts[alert_key]
                return alert
            
            return None
    
    def check_threshold(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        operator: str = ">",
        level: AlertLevel = AlertLevel.WARNING,
    ) -> Optional[Alert]:
        """
        Check if metric exceeds threshold.
        
        Args:
            metric_name: Metric name
            value: Current value
            threshold: Threshold value
            operator: Comparison operator (>, <, >=, <=, ==)
            level: Alert level
        
        Returns:
            Alert if threshold exceeded, None otherwise
        """
        exceeded = False
        
        if operator == ">":
            exceeded = value > threshold
        elif operator == "<":
            exceeded = value < threshold
        elif operator == ">=":
            exceeded = value >= threshold
        elif operator == "<=":
            exceeded = value <= threshold
        elif operator == "==":
            exceeded = value == threshold
        else:
            logger.warning(f"Unknown operator: {operator}")
            return None
        
        if exceeded:
            alert = Alert(
                level=level,
                message=f"{metric_name} {operator} {threshold} (current: {value:.3f})",
                metric=metric_name,
                value=value,
                threshold=threshold,
                condition=AlertCondition.THRESHOLD,
            )
            self.trigger_alert(alert)
            return alert
        
        # Resolve alert if threshold no longer exceeded
        self.resolve_alert(metric_name, level)
        return None
    
    def check_rate_of_change(
        self,
        metric_name: str,
        current_value: float,
        previous_value: float,
        threshold_percent: float,
        level: AlertLevel = AlertLevel.WARNING,
    ) -> Optional[Alert]:
        """
        Check rate of change in metric.
        
        Args:
            metric_name: Metric name
            current_value: Current value
            previous_value: Previous value
            threshold_percent: Threshold percentage change
            level: Alert level
        
        Returns:
            Alert if rate of change exceeds threshold
        """
        if previous_value == 0:
            return None
        
        change_percent = abs((current_value - previous_value) / previous_value) * 100.0
        
        if change_percent > threshold_percent:
            alert = Alert(
                level=level,
                message=(
                    f"{metric_name} changed by {change_percent:.1f}% "
                    f"(from {previous_value:.3f} to {current_value:.3f})"
                ),
                metric=metric_name,
                value=current_value,
                threshold=threshold_percent,
                condition=AlertCondition.RATE_OF_CHANGE,
                metadata={
                    "previous_value": previous_value,
                    "change_percent": change_percent,
                },
            )
            self.trigger_alert(alert)
            return alert
        
        return None
    
    def get_recent_alerts(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 100,
        unresolved_only: bool = False,
    ) -> List[Alert]:
        """
        Get recent alerts.
        
        Args:
            level: Filter by level (optional)
            limit: Maximum number of alerts
            unresolved_only: Only return unresolved alerts
        
        Returns:
            List of recent alerts
        """
        with self.lock:
            alerts = self.alerts[-limit:] if limit else self.alerts
            
            if unresolved_only:
                alerts = [a for a in alerts if not a.resolved]
            
            if level:
                alerts = [a for a in alerts if a.level == level]
            
            return alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        with self.lock:
            total = len(self.alerts)
            active = len(self.active_alerts)
            resolved = total - active
            
            by_level = {}
            for level in AlertLevel:
                count = sum(1 for a in self.alerts if a.level == level)
                active_count = sum(1 for a in self.active_alerts.values() if a.level == level)
                by_level[level.value] = {
                    "total": count,
                    "active": active_count,
                    "resolved": count - active_count,
                }
            
            return {
                "total": total,
                "active": active,
                "resolved": resolved,
                "by_level": by_level,
            }


class HealthMonitor:
    """
    Enhanced system health monitor.
    
    Features:
    - Comprehensive health checks
    - Component-specific health
    - Health history
    - Automatic alerting
    """
    
    def __init__(self):
        """Initialize health monitor."""
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.health_status: Dict[str, Any] = {}
        self.health_history: deque = deque(maxlen=1000)
        self.component_health: Dict[str, Dict[str, Any]] = {}
    
    def check_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "components": {},
            "alerts": self.alert_manager.get_alert_summary(),
        }
        
        # Check metrics
        all_metrics = self.metric_collector.get_all_metrics()
        for name, values in all_metrics.items():
            stats = self.metric_collector.get_metric_stats(name)
            if stats:
                health["metrics"][name] = stats
        
        # Check component health
        for component, component_health in self.component_health.items():
            health["components"][component] = component_health
        
        # Determine overall health
        alert_summary = health["alerts"]
        
        if alert_summary["active"] > 0:
            # Check for critical alerts
            critical_count = alert_summary["by_level"].get("critical", {}).get("active", 0)
            if critical_count > 0:
                health["status"] = "unhealthy"
            else:
                error_count = alert_summary["by_level"].get("error", {}).get("active", 0)
                if error_count > 0:
                    health["status"] = "degraded"
                else:
                    health["status"] = "warning"
        
        # Store in history
        self.health_history.append(health)
        self.health_status = health
        
        return health
    
    def set_component_health(
        self,
        component: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set health status for a component.
        
        Args:
            component: Component name
            status: Health status (healthy, degraded, unhealthy)
            details: Optional details dictionary
        """
        self.component_health[component] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }
    
    def monitor_metric(
        self,
        name: str,
        value: float,
        thresholds: Optional[Dict[str, Tuple[float, str, AlertLevel]]] = None,
    ) -> None:
        """
        Monitor a metric with automatic alerting.
        
        Args:
            name: Metric name
            value: Current value
            thresholds: Optional thresholds dict {operator: (threshold, level)}
        """
        self.metric_collector.record_metric(name, value)
        
        if thresholds:
            for operator, (threshold, level) in thresholds.items():
                self.alert_manager.check_threshold(
                    name, value, threshold, operator, level
                )
    
    def get_health_history(
        self,
        time_window: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get health history.
        
        Args:
            time_window: Optional time window in seconds
        
        Returns:
            List of health status dictionaries
        """
        if time_window:
            cutoff = time.time() - time_window
            return [
                h for h in self.health_history
                if datetime.fromisoformat(h["timestamp"]).timestamp() >= cutoff
            ]
        else:
            return list(self.health_history)


# Global monitoring instances
metric_collector = MetricCollector()
alert_manager = AlertManager()
health_monitor = HealthMonitor()


__all__ = [
    "AlertLevel",
    "AlertCondition",
    "Alert",
    "MetricSnapshot",
    "MetricCollector",
    "AlertManager",
    "HealthMonitor",
    "metric_collector",
    "alert_manager",
    "health_monitor",
]
