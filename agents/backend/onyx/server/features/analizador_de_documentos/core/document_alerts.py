"""
Document Alerts - Sistema de Alertas Inteligente
================================================

Sistema de alertas para notificaciones automáticas.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severidad de alerta."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """Regla de alerta."""
    rule_id: str
    name: str
    condition: Callable[[Any], bool]
    severity: AlertSeverity
    message: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutos


@dataclass
class Alert:
    """Alerta."""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


class AlertManager:
    """Gestor de alertas."""
    
    def __init__(self, analyzer):
        """Inicializar gestor."""
        self.analyzer = analyzer
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.last_triggered: Dict[str, datetime] = {}
        self.alert_handlers: List[Callable] = []
        self.max_alerts = 1000
    
    def register_rule(self, rule: AlertRule):
        """Registrar regla de alerta."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Regla de alerta registrada: {rule.name}")
    
    def register_handler(self, handler: Callable):
        """Registrar handler de alertas."""
        self.alert_handlers.append(handler)
    
    async def check_alerts(self, context: Dict[str, Any]):
        """Verificar reglas de alerta."""
        triggered_alerts = []
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Verificar cooldown
            if rule_id in self.last_triggered:
                elapsed = (datetime.now() - self.last_triggered[rule_id]).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    continue
            
            # Evaluar condición
            try:
                if rule.condition(context):
                    alert = Alert(
                        alert_id=f"alert_{len(self.alerts) + 1}",
                        rule_id=rule_id,
                        severity=rule.severity,
                        message=rule.message,
                        data=context
                    )
                    
                    self.alerts.append(alert)
                    self.last_triggered[rule_id] = datetime.now()
                    triggered_alerts.append(alert)
                    
                    # Notificar handlers
                    for handler in self.alert_handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(alert)
                            else:
                                handler(alert)
                        except Exception as e:
                            logger.error(f"Error en handler de alerta: {e}")
                    
                    logger.warning(f"Alerta disparada: {rule.name} - {rule.message}")
            except Exception as e:
                logger.error(f"Error evaluando regla {rule_id}: {e}")
        
        # Mantener solo últimos N alertas
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        return triggered_alerts
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        unacknowledged_only: bool = True
    ) -> List[Alert]:
        """Obtener alertas activas."""
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str):
        """Reconocer alerta."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alerta {alert_id} reconocida")
                return True
        return False
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de alertas."""
        total = len(self.alerts)
        by_severity = {
            severity.value: len([a for a in self.alerts if a.severity == severity])
            for severity in AlertSeverity
        }
        unacknowledged = len([a for a in self.alerts if not a.acknowledged])
        
        return {
            "total_alerts": total,
            "by_severity": by_severity,
            "unacknowledged": unacknowledged,
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules.values() if r.enabled])
        }


# Reglas predefinidas
def create_quality_alert_rule(threshold: float = 50.0) -> AlertRule:
    """Crear regla de alerta para calidad baja."""
    return AlertRule(
        rule_id="low_quality",
        name="Calidad Baja",
        condition=lambda ctx: ctx.get("quality_score", 100) < threshold,
        severity=AlertSeverity.HIGH,
        message=f"Calidad del documento por debajo de {threshold}",
        cooldown_seconds=600
    )


def create_grammar_alert_rule(threshold: float = 60.0) -> AlertRule:
    """Crear regla de alerta para gramática baja."""
    return AlertRule(
        rule_id="low_grammar",
        name="Gramática Baja",
        condition=lambda ctx: ctx.get("grammar_score", 100) < threshold,
        severity=AlertSeverity.MEDIUM,
        message=f"Gramática del documento por debajo de {threshold}",
        cooldown_seconds=600
    )


def create_processing_time_alert_rule(threshold: float = 10.0) -> AlertRule:
    """Crear regla de alerta para tiempo de procesamiento alto."""
    return AlertRule(
        rule_id="high_processing_time",
        name="Tiempo de Procesamiento Alto",
        condition=lambda ctx: ctx.get("processing_time", 0) > threshold,
        severity=AlertSeverity.MEDIUM,
        message=f"Tiempo de procesamiento excede {threshold}s",
        cooldown_seconds=300
    )


__all__ = [
    "AlertManager",
    "AlertRule",
    "Alert",
    "AlertSeverity",
    "create_quality_alert_rule",
    "create_grammar_alert_rule",
    "create_processing_time_alert_rule"
]
















