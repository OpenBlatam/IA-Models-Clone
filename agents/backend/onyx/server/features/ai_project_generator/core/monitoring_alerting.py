"""
Monitoring and Alerting - Monitoreo y Alertas Avanzado
======================================================

Sistema avanzado de monitoreo y alertas:
- Custom metrics
- Alert rules
- Alert channels
- Threshold management
- Alert aggregation
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Canales de alerta"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SMS = "sms"


class AlertRule:
    """Regla de alerta"""
    
    def __init__(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        condition: str = ">",
        severity: AlertSeverity = AlertSeverity.WARNING,
        duration: int = 60
    ) -> None:
        self.name = name
        self.metric_name = metric_name
        self.threshold = threshold
        self.condition = condition
        self.severity = severity
        self.duration = duration  # segundos
        self.enabled = True
    
    def evaluate(self, value: float) -> bool:
        """Evalúa si la alerta debe dispararse"""
        if not self.enabled:
            return False
        
        if self.condition == ">":
            return value > self.threshold
        elif self.condition == ">=":
            return value >= self.threshold
        elif self.condition == "<":
            return value < self.threshold
        elif self.condition == "<=":
            return value <= self.threshold
        elif self.condition == "==":
            return value == self.threshold
        else:
            return False


class Alert:
    """Alerta"""
    
    def __init__(
        self,
        rule_name: str,
        metric_name: str,
        value: float,
        severity: AlertSeverity,
        message: str
    ) -> None:
        self.rule_name = rule_name
        self.metric_name = metric_name
        self.value = value
        self.severity = severity
        self.message = message
        self.timestamp = datetime.now()
        self.acknowledged = False
        self.resolved = False


class MonitoringSystem:
    """
    Sistema de monitoreo y alertas.
    """
    
    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_channels: Dict[AlertChannel, Callable] = {}
        self.metric_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Registra métrica"""
        if name not in self.metrics:
            self.metrics[name] = []
            self.metric_history[name] = []
        
        self.metrics[name].append(value)
        
        # Mantener solo últimos 1000 valores
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
        
        # Guardar en historial
        self.metric_history[name].append({
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {}
        })
        
        # Evaluar reglas de alerta
        self._evaluate_alerts(name, value)
    
    def _evaluate_alerts(self, metric_name: str, value: float) -> None:
        """Evalúa reglas de alerta"""
        for rule_name, rule in self.alert_rules.items():
            if rule.metric_name == metric_name and rule.evaluate(value):
                # Verificar si ya existe alerta activa
                if rule_name not in self.active_alerts:
                    alert = Alert(
                        rule_name=rule_name,
                        metric_name=metric_name,
                        value=value,
                        severity=rule.severity,
                        message=f"{rule.name}: {metric_name} = {value} {rule.condition} {rule.threshold}"
                    )
                    self.active_alerts[rule_name] = alert
                    self._send_alert(alert)
    
    def register_alert_rule(self, rule: AlertRule) -> None:
        """Registra regla de alerta"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Alert rule registered: {rule.name}")
    
    def register_alert_channel(
        self,
        channel: AlertChannel,
        handler: Callable[[Alert], None]
    ) -> None:
        """Registra canal de alerta"""
        self.alert_channels[channel] = handler
        logger.info(f"Alert channel registered: {channel.value}")
    
    def _send_alert(self, alert: Alert) -> None:
        """Envía alerta a canales configurados"""
        for channel, handler in self.alert_channels.items():
            try:
                handler(alert)
                logger.info(f"Alert sent via {channel.value}: {alert.message}")
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
    
    def get_metric_stats(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas de una métrica"""
        if metric_name not in self.metrics:
            return None
        
        values = self.metrics[metric_name]
        if not values:
            return None
        
        return {
            "name": metric_name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1]
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Obtiene alertas activas"""
        return [
            {
                "rule_name": alert.rule_name,
                "metric": alert.metric_name,
                "value": alert.value,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in self.active_alerts.values()
            if not alert.resolved
        ]
    
    def acknowledge_alert(self, rule_name: str) -> None:
        """Reconoce una alerta"""
        if rule_name in self.active_alerts:
            self.active_alerts[rule_name].acknowledged = True


def get_monitoring_system() -> MonitoringSystem:
    """Obtiene sistema de monitoreo"""
    return MonitoringSystem()















