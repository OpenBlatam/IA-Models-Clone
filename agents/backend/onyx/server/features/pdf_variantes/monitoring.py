"""
Monitoring System
=================

Monitoring and metrics collection for PDF Variantes.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """A metric."""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Alert:
    """An alert."""
    alert_id: str
    name: str
    message: str
    severity: str  # "critical", "warning", "info"
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "message": self.message,
            "severity": self.severity,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "triggered_at": self.triggered_at.isoformat()
        }


class MonitoringSystem:
    """Monitoring and alerting system."""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Callable] = {}
        self.lock = threading.Lock()
        logger.info("Initialized Monitoring System")
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric."""
        
        with self.lock:
            metric = Metric(
                name=name,
                type=metric_type,
                value=value,
                labels=labels or {}
            )
            
            key = f"{name}_{str(labels)}"
            self.metrics[key].append(metric)
            
            # Check alert rules
            self._check_alert_rules(name, value)
    
    def record_counter(
        self,
        name: str,
        increment: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a counter metric."""
        
        self.record_metric(name, increment, MetricType.COUNTER, labels)
    
    def get_metric(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        window_minutes: Optional[int] = None
    ) -> List[Metric]:
        """Get metrics for a name."""
        
        with self.lock:
            key = f"{name}_{str(labels)}"
            metrics = list(self.metrics[key])
            
            if window_minutes:
                cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
                metrics = [m for m in metrics if m.timestamp >= cutoff]
            
            return metrics
    
    def get_latest_metric(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[Metric]:
        """Get latest metric value."""
        
        metrics = self.get_metric(name, labels)
        
        if metrics:
            return max(metrics, key=lambda m: m.timestamp)
        
        return None
    
    def get_metric_summary(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get metric summary statistics."""
        
        metrics = self.get_metric(name, labels, window_minutes)
        
        if not metrics:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "sum": None
            }
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values)
        }
    
    def add_alert_rule(
        self,
        name: str,
        threshold: float,
        condition: str,
        severity: str = "warning",
        message: Optional[str] = None
    ):
        """Add an alert rule."""
        
        def check_condition(metric_value: float) -> bool:
            if condition == "greater_than":
                return metric_value > threshold
            elif condition == "less_than":
                return metric_value < threshold
            elif condition == "equals":
                return metric_value == threshold
            else:
                return False
        
        self.alert_rules[name] = {
            "threshold": threshold,
            "condition": check_condition,
            "severity": severity,
            "message": message or f"Alert: {name} crossed threshold"
        }
        
        logger.info(f"Added alert rule: {name}")
    
    def _check_alert_rules(self, metric_name: str, metric_value: float):
        """Check alert rules against metric value."""
        
        for name, rule in self.alert_rules.items():
            # Only check if this metric matches
            if rule.get("metric_name") == metric_name:
                if rule["condition"](metric_value):
                    self._trigger_alert(name, rule, metric_value)
    
    def _trigger_alert(self, alert_name: str, rule: Dict[str, Any], current_value: float):
        """Trigger an alert."""
        
        alert = Alert(
            alert_id=f"alert_{datetime.utcnow().timestamp()}",
            name=alert_name,
            message=rule["message"],
            severity=rule["severity"],
            threshold=rule["threshold"],
            current_value=current_value
        )
        
        with self.lock:
            self.alerts.append(alert)
        
        logger.warning(f"Alert triggered: {alert_name} ({current_value})")
    
    def get_active_alerts(
        self,
        severity: Optional[str] = None,
        max_alerts: int = 100
    ) -> List[Alert]:
        """Get active alerts."""
        
        with self.lock:
            alerts = self.alerts[-max_alerts:]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            return alerts
    
    def clear_alerts(self):
        """Clear all alerts."""
        
        with self.lock:
            self.alerts.clear()
        
        logger.info("Cleared all alerts")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        
        with self.lock:
            critical_alerts = [a for a in self.alerts if a.severity == "critical"]
            warning_alerts = [a for a in self.alerts if a.severity == "warning"]
            
            return {
                "status": "healthy" if len(critical_alerts) == 0 else "unhealthy",
                "critical_alerts": len(critical_alerts),
                "warning_alerts": len(warning_alerts),
                "total_alerts": len(self.alerts),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        
        with self.lock:
            summary = {}
            
            for key, metrics_queue in self.metrics.items():
                if metrics_queue:
                    values = [m.value for m in metrics_queue]
                    
                    summary[key] = {
                        "count": len(values),
                        "latest": values[-1],
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values)
                    }
            
            return summary
