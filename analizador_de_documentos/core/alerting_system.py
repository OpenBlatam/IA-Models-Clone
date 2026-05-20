"""
Sistema de Alertas Avanzado
============================

Sistema para alertas configurable con reglas personalizadas.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severidad de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """Condiciones de alerta"""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    TREND_INCREASING = "trend_increasing"
    TREND_DECREASING = "trend_decreasing"


@dataclass
class AlertRule:
    """Regla de alerta"""
    name: str
    description: str
    metric: str
    condition: AlertCondition
    threshold: Any
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 60
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """Alerta generada"""
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_value: Any
    threshold: Any
    timestamp: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class AlertingSystem:
    """
    Sistema de alertas avanzado
    
    Proporciona:
    - Alertas configurables
    - Múltiples condiciones
    - Cooldown periods
    - Historial de alertas
    - Integración con notificaciones
    """
    
    def __init__(self):
        """Inicializar sistema de alertas"""
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: deque = deque(maxlen=1000)
        logger.info("AlertingSystem inicializado")
    
    def add_rule(
        self,
        name: str,
        description: str,
        metric: str,
        condition: AlertCondition,
        threshold: Any,
        severity: AlertSeverity,
        cooldown_minutes: int = 60
    ) -> AlertRule:
        """
        Agregar regla de alerta
        
        Args:
            name: Nombre de la regla
            description: Descripción
            metric: Nombre de la métrica a monitorear
            condition: Condición de alerta
            threshold: Umbral
            severity: Severidad
            cooldown_minutes: Minutos de cooldown entre alertas
        
        Returns:
            AlertRule creada
        """
        rule = AlertRule(
            name=name,
            description=description,
            metric=metric,
            condition=condition,
            threshold=threshold,
            severity=severity,
            cooldown_minutes=cooldown_minutes
        )
        
        self.rules[name] = rule
        logger.info(f"Regla de alerta agregada: {name}")
        
        return rule
    
    def check_alerts(
        self,
        metrics: Dict[str, Any]
    ) -> List[Alert]:
        """
        Verificar alertas basadas en métricas
        
        Args:
            metrics: Diccionario con valores de métricas
        
        Returns:
            Lista de alertas generadas
        """
        alerts = []
        now = datetime.now()
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Verificar cooldown
            if rule.last_triggered:
                time_since_last = now - rule.last_triggered
                if time_since_last < timedelta(minutes=rule.cooldown_minutes):
                    continue
            
            # Obtener valor de métrica
            metric_value = metrics.get(rule.metric)
            if metric_value is None:
                continue
            
            # Verificar condición
            should_alert = self._check_condition(
                metric_value,
                rule.condition,
                rule.threshold
            )
            
            if should_alert:
                alert = Alert(
                    rule_name=rule_name,
                    severity=rule.severity,
                    message=f"{rule.description}: {rule.metric} = {metric_value}",
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    metadata={"rule": rule.description}
                )
                
                alerts.append(alert)
                self.alert_history.append(alert)
                rule.last_triggered = now
        
        return alerts
    
    def _check_condition(
        self,
        value: Any,
        condition: AlertCondition,
        threshold: Any
    ) -> bool:
        """Verificar condición de alerta"""
        try:
            if condition == AlertCondition.GREATER_THAN:
                return float(value) > float(threshold)
            elif condition == AlertCondition.LESS_THAN:
                return float(value) < float(threshold)
            elif condition == AlertCondition.EQUALS:
                return value == threshold
            elif condition == AlertCondition.NOT_EQUALS:
                return value != threshold
            elif condition == AlertCondition.CONTAINS:
                return str(threshold) in str(value)
            elif condition == AlertCondition.NOT_CONTAINS:
                return str(threshold) not in str(value)
            else:
                return False
        except (ValueError, TypeError):
            return False
    
    def get_alert_history(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Obtener historial de alertas"""
        history = list(self.alert_history)
        
        if severity:
            history = [a for a in history if a.severity == severity]
        
        return history[-limit:]
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Obtener lista de reglas"""
        return [
            {
                "name": r.name,
                "description": r.description,
                "metric": r.metric,
                "condition": r.condition.value,
                "threshold": r.threshold,
                "severity": r.severity.value,
                "enabled": r.enabled,
                "cooldown_minutes": r.cooldown_minutes
            }
            for r in self.rules.values()
        ]


# Instancia global
_alerting_system: Optional[AlertingSystem] = None


def get_alerting_system() -> AlertingSystem:
    """Obtener instancia global del sistema de alertas"""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
    return _alerting_system
