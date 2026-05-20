"""
Alerting System Service - Sistema de alertas
=============================================

Sistema de alertas y notificaciones para eventos críticos.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Severidad de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Estados de alerta"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Regla de alerta"""
    id: str
    name: str
    condition: Callable  # Función que evalúa condición
    severity: AlertSeverity
    message: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutos por defecto
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """Alerta"""
    id: str
    rule_id: str
    severity: AlertSeverity
    message: str
    status: AlertStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class AlertingSystemService:
    """Servicio de sistema de alertas"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = {}
        logger.info("AlertingSystemService initialized")
    
    def create_alert_rule(
        self,
        name: str,
        condition: Callable,
        severity: AlertSeverity,
        message: str,
        cooldown_seconds: int = 300
    ) -> AlertRule:
        """Crear regla de alerta"""
        rule_id = f"rule_{name.lower().replace(' ', '_')}"
        
        rule = AlertRule(
            id=rule_id,
            name=name,
            condition=condition,
            severity=severity,
            message=message,
            cooldown_seconds=cooldown_seconds,
        )
        
        self.rules[rule_id] = rule
        
        logger.info(f"Alert rule created: {rule_id}")
        return rule
    
    def register_handler(
        self,
        severity: AlertSeverity,
        handler: Callable
    ):
        """Registrar handler para severidad"""
        if severity not in self.alert_handlers:
            self.alert_handlers[severity] = []
        
        self.alert_handlers[severity].append(handler)
    
    def check_conditions(self, context: Dict[str, Any]) -> List[Alert]:
        """Verificar condiciones y generar alertas"""
        triggered_alerts = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Verificar cooldown
            if rule.last_triggered:
                time_since_last = datetime.now() - rule.last_triggered
                if time_since_last.total_seconds() < rule.cooldown_seconds:
                    continue
            
            # Evaluar condición
            try:
                if rule.condition(context):
                    alert = self._create_alert(rule, context)
                    triggered_alerts.append(alert)
                    rule.last_triggered = datetime.now()
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.id}: {e}")
        
        return triggered_alerts
    
    def _create_alert(self, rule: AlertRule, context: Dict[str, Any]) -> Alert:
        """Crear alerta"""
        alert_id = f"alert_{int(datetime.now().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            severity=rule.severity,
            message=rule.message,
            status=AlertStatus.ACTIVE,
            metadata=context,
        )
        
        self.alerts[alert_id] = alert
        
        # Ejecutar handlers
        handlers = self.alert_handlers.get(rule.severity, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.warning(f"Alert triggered: {alert_id} - {rule.message}")
        return alert
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Reconocer alerta"""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.metadata["acknowledged_by"] = user_id
        
        return True
    
    def resolve_alert(self, alert_id: str, user_id: str) -> bool:
        """Resolver alerta"""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.metadata["resolved_by"] = user_id
        
        return True
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> List[Dict[str, Any]]:
        """Obtener alertas activas"""
        active_alerts = [
            a for a in self.alerts.values()
            if a.status == AlertStatus.ACTIVE
            and (severity is None or a.severity == severity)
        ]
        
        return [
            {
                "id": a.id,
                "rule_id": a.rule_id,
                "severity": a.severity.value,
                "message": a.message,
                "created_at": a.created_at.isoformat(),
                "metadata": a.metadata,
            }
            for a in sorted(active_alerts, key=lambda x: x.created_at, reverse=True)
        ]
    
    def get_alert_statistics(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Obtener estadísticas de alertas"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            a for a in self.alerts.values()
            if a.created_at >= cutoff_time
        ]
        
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "period_hours": hours,
            "total_alerts": len(recent_alerts),
            "active_alerts": sum(1 for a in recent_alerts if a.status == AlertStatus.ACTIVE),
            "resolved_alerts": sum(1 for a in recent_alerts if a.status == AlertStatus.RESOLVED),
            "severity_breakdown": severity_counts,
        }




