"""
Monitoring System
================

Sistema de monitoreo y alertas.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .metrics import get_metrics_collector, record_value

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Niveles de alerta."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alerta del sistema."""
    level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertRule:
    """
    Regla de alerta.
    
    Define condiciones para generar alertas.
    """
    
    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        level: AlertLevel = AlertLevel.WARNING,
        message_template: str = "{name} condition met"
    ):
        """
        Inicializar regla de alerta.
        
        Args:
            name: Nombre de la regla
            condition: Función que evalúa la condición
            level: Nivel de alerta
            message_template: Plantilla del mensaje
        """
        self.name = name
        self.condition = condition
        self.level = level
        self.message_template = message_template
        self.enabled = True
    
    def check(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        """
        Verificar condición.
        
        Args:
            metrics: Métricas actuales
            
        Returns:
            Alert si la condición se cumple, None en caso contrario
        """
        if not self.enabled:
            return None
        
        try:
            if self.condition(metrics):
                message = self.message_template.format(
                    name=self.name,
                    **metrics
                )
                return Alert(
                    level=self.level,
                    message=message,
                    source=self.name
                )
        except Exception as e:
            logger.error(f"Error checking alert rule {self.name}: {e}")
        
        return None


class MonitoringSystem:
    """
    Sistema de monitoreo y alertas.
    
    Monitorea métricas y genera alertas según reglas.
    """
    
    def __init__(self):
        """Inicializar sistema de monitoreo."""
        self.rules: List[AlertRule] = []
        self.alerts: List[Alert] = []
        self.max_alerts = 1000
        self.callbacks: List[Callable[[Alert], None]] = []
        self.metrics_collector = get_metrics_collector()
    
    def add_rule(self, rule: AlertRule) -> None:
        """Agregar regla de alerta."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Agregar callback para alertas."""
        self.callbacks.append(callback)
    
    def check_rules(self) -> List[Alert]:
        """
        Verificar todas las reglas.
        
        Returns:
            Lista de alertas generadas
        """
        metrics = self.metrics_collector.get_all_metrics()
        new_alerts = []
        
        for rule in self.rules:
            alert = rule.check(metrics)
            if alert:
                new_alerts.append(alert)
                self._handle_alert(alert)
        
        return new_alerts
    
    def _handle_alert(self, alert: Alert) -> None:
        """Manejar alerta."""
        # Agregar a historial
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)
        
        # Registrar métrica
        record_value("monitoring.alerts", 1.0, {
            "level": alert.level.value,
            "source": alert.source
        })
        
        # Log según nivel
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(f"ALERT [{alert.source}]: {alert.message}")
        elif alert.level == AlertLevel.ERROR:
            logger.error(f"ALERT [{alert.source}]: {alert.message}")
        elif alert.level == AlertLevel.WARNING:
            logger.warning(f"ALERT [{alert.source}]: {alert.message}")
        else:
            logger.info(f"ALERT [{alert.source}]: {alert.message}")
        
        # Ejecutar callbacks
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_recent_alerts(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 10
    ) -> List[Alert]:
        """
        Obtener alertas recientes.
        
        Args:
            level: Filtrar por nivel (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de alertas
        """
        alerts = self.alerts[-limit:] if limit else self.alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de alertas."""
        total = len(self.alerts)
        by_level = {}
        
        for level in AlertLevel:
            count = sum(1 for a in self.alerts if a.level == level)
            by_level[level.value] = count
        
        return {
            "total_alerts": total,
            "by_level": by_level,
            "active_rules": len([r for r in self.rules if r.enabled])
        }


# Instancia global
_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> MonitoringSystem:
    """Obtener instancia global del sistema de monitoreo."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system


def create_threshold_rule(
    metric_name: str,
    threshold: float,
    comparison: str = ">",
    level: AlertLevel = AlertLevel.WARNING
) -> AlertRule:
    """
    Crear regla de umbral.
    
    Args:
        metric_name: Nombre de la métrica
        threshold: Valor umbral
        comparison: Comparación (">", "<", ">=", "<=", "==")
        level: Nivel de alerta
        
    Returns:
        AlertRule configurada
    """
    def condition(metrics: Dict[str, Any]) -> bool:
        metric = metrics.get(metric_name)
        if metric is None:
            return False
        
        value = metric.get("latest") or metric.get("average") or 0.0
        
        if comparison == ">":
            return value > threshold
        elif comparison == "<":
            return value < threshold
        elif comparison == ">=":
            return value >= threshold
        elif comparison == "<=":
            return value <= threshold
        elif comparison == "==":
            return abs(value - threshold) < 1e-6
        
        return False
    
    return AlertRule(
        name=f"{metric_name}_{comparison}_{threshold}",
        condition=condition,
        level=level,
        message_template=f"{metric_name} is {comparison} {threshold} (current: {{latest}})"
    )






