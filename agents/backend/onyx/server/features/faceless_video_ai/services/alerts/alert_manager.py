"""
Alert Manager
Advanced alerting system
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict

from .alert_rules import AlertRule, AlertSeverity

logger = logging.getLogger(__name__)


class Alert:
    """Represents an alert"""
    
    def __init__(
        self,
        alert_id: str,
        rule_name: str,
        severity: AlertSeverity,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.alert_id = alert_id
        self.rule_name = rule_name
        self.severity = severity
        self.message = message
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.acknowledged = False
        self.resolved = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


class AlertManager:
    """Manages alerts and alerting rules"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default alert rules"""
        # High error rate
        self.register_rule(AlertRule(
            name="high_error_rate",
            condition=lambda metrics: metrics.get("error_rate", 0) > 0.1,
            severity=AlertSeverity.CRITICAL,
            message="Error rate exceeds 10%"
        ))
        
        # Low cache hit rate
        self.register_rule(AlertRule(
            name="low_cache_hit_rate",
            condition=lambda metrics: metrics.get("cache_hit_rate", 1.0) < 0.5,
            severity=AlertSeverity.WARNING,
            message="Cache hit rate below 50%"
        ))
        
        # High queue size
        self.register_rule(AlertRule(
            name="high_queue_size",
            condition=lambda metrics: metrics.get("queue_size", 0) > 100,
            severity=AlertSeverity.WARNING,
            message="Video generation queue size exceeds 100"
        ))
        
        # Slow API response
        self.register_rule(AlertRule(
            name="slow_api_response",
            condition=lambda metrics: metrics.get("avg_api_duration", 0) > 5.0,
            severity=AlertSeverity.WARNING,
            message="Average API response time exceeds 5 seconds"
        ))
    
    def register_rule(self, rule: AlertRule):
        """Register alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")
    
    def register_handler(self, severity: AlertSeverity, handler: Callable):
        """Register alert handler"""
        self.alert_handlers[severity].append(handler)
        logger.info(f"Registered alert handler for {severity.value}")
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against metrics"""
        for rule_name, rule in self.rules.items():
            try:
                if rule.condition(metrics):
                    self._trigger_alert(rule, metrics)
            except Exception as e:
                logger.error(f"Error checking rule {rule_name}: {str(e)}")
    
    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert"""
        alert_id = f"alert_{len(self.alerts) + 1}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=rule.message,
            metadata={"metrics": metrics}
        )
        
        self.alerts[alert_id] = alert
        
        # Call handlers
        for handler in self.alert_handlers[rule.severity]:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {str(e)}")
        
        logger.warning(f"Alert triggered: {rule.name} - {rule.message}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active (unresolved) alerts"""
        alerts = [a for a in self.alerts.values() if not a.resolved]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge alert"""
        alert = self.alerts.get(alert_id)
        if alert:
            alert.acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged")
    
    def resolve_alert(self, alert_id: str):
        """Resolve alert"""
        alert = self.alerts.get(alert_id)
        if alert:
            alert.resolved = True
            logger.info(f"Alert {alert_id} resolved")


_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get alert manager instance (singleton)"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager

