"""
Sistema de Alertas Avanzado para Suno Clone AI

Proporciona:
- Alertas basadas en umbrales
- Notificaciones en tiempo real
- Integración con WebSocket
- Historial de alertas
- Configuración dinámica de umbrales
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


@dataclass
class Alert:
    """Representa una alerta"""
    level: AlertLevel
    type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la alerta a diccionario para serialización"""
        return {
            "level": self.level.value,
            "type": self.type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class AlertRule:
    """Regla de alerta con umbrales configurables"""
    
    def __init__(
        self,
        name: str,
        alert_type: str,
        threshold: float,
        level: AlertLevel = AlertLevel.WARNING,
        condition: str = "greater_than",
        cooldown_seconds: int = 300
    ):
        """
        Args:
            name: Nombre de la regla
            alert_type: Tipo de alerta (memory, cpu, errors, etc.)
            threshold: Umbral para activar la alerta
            level: Nivel de alerta
            condition: Condición ("greater_than", "less_than", "equals")
            cooldown_seconds: Tiempo mínimo entre alertas del mismo tipo
        """
        self.name = name
        self.alert_type = alert_type
        self.threshold = threshold
        self.level = level
        self.condition = condition
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered: Optional[datetime] = None
    
    def check(self, value: float) -> bool:
        """Verifica si el valor activa la alerta"""
        # Verificar cooldown
        if self.last_triggered:
            time_since = (datetime.now() - self.last_triggered).total_seconds()
            if time_since < self.cooldown_seconds:
                return False
        
        # Verificar condición
        if self.condition == "greater_than":
            triggered = value > self.threshold
        elif self.condition == "less_than":
            triggered = value < self.threshold
        elif self.condition == "equals":
            triggered = value == self.threshold
        else:
            triggered = False
        
        if triggered:
            self.last_triggered = datetime.now()
        
        return triggered


class AlertManager:
    """Gestor de alertas del sistema"""
    
    def __init__(self, max_history: int = 1000):
        """
        Args:
            max_history: Número máximo de alertas en el historial
        """
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=max_history)
        self.callbacks: List[Callable[[Alert], None]] = []
        
        # Reglas por defecto
        self._setup_default_rules()
        
        logger.info("AlertManager initialized")
    
    def _setup_default_rules(self):
        """Configura reglas de alerta por defecto"""
        self.add_rule(AlertRule(
            name="high_memory",
            alert_type="memory",
            threshold=90.0,
            level=AlertLevel.WARNING,
            condition="greater_than"
        ))
        
        self.add_rule(AlertRule(
            name="critical_memory",
            alert_type="memory",
            threshold=95.0,
            level=AlertLevel.CRITICAL,
            condition="greater_than"
        ))
        
        self.add_rule(AlertRule(
            name="high_cpu",
            alert_type="cpu",
            threshold=90.0,
            level=AlertLevel.WARNING,
            condition="greater_than"
        ))
        
        self.add_rule(AlertRule(
            name="high_error_count",
            alert_type="errors",
            threshold=50.0,
            level=AlertLevel.WARNING,
            condition="greater_than"
        ))
        
        self.add_rule(AlertRule(
            name="critical_error_count",
            alert_type="errors",
            threshold=100.0,
            level=AlertLevel.CRITICAL,
            condition="greater_than"
        ))
        
        self.add_rule(AlertRule(
            name="slow_generation",
            alert_type="generation_time",
            threshold=60.0,
            level=AlertLevel.WARNING,
            condition="greater_than"
        ))
    
    def add_rule(self, rule: AlertRule):
        """Agrega una regla de alerta"""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Elimina una regla de alerta"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def register_callback(self, callback: Callable[[Alert], None]):
        """Registra un callback para cuando se active una alerta"""
        self.callbacks.append(callback)
    
    def check_metrics(self, metrics: Dict[str, Any]):
        """
        Verifica métricas contra las reglas de alerta.
        
        Args:
            metrics: Diccionario con métricas del sistema
        """
        alerts_triggered = []
        
        # Verificar reglas
        for rule_name, rule in self.rules.items():
            value = self._get_metric_value(metrics, rule.alert_type)
            if value is not None and rule.check(value):
                alert = self._create_alert(rule, value, metrics)
                alerts_triggered.append(alert)
        
        # Procesar alertas
        for alert in alerts_triggered:
            self._process_alert(alert)
        
        return alerts_triggered
    
    def _get_metric_value(self, metrics: Dict[str, Any], alert_type: str) -> Optional[float]:
        """Extrae el valor de métrica según el tipo de alerta"""
        if alert_type == "memory":
            return metrics.get("system_info", {}).get("memory_percent")
        elif alert_type == "cpu":
            return metrics.get("system_info", {}).get("cpu_percent")
        elif alert_type == "errors":
            error_counts = metrics.get("generation_metrics", {}).get("error_counts", {})
            return float(sum(error_counts.values()))
        elif alert_type == "generation_time":
            return metrics.get("generation_metrics", {}).get("avg_generation_time_seconds")
        return None
    
    def _create_alert(self, rule: AlertRule, value: float, metrics: Dict[str, Any]) -> Alert:
        """Crea una alerta desde una regla"""
        message = f"{rule.alert_type}: {value:.2f} exceeds threshold {rule.threshold}"
        
        return Alert(
            level=rule.level,
            type=rule.alert_type,
            message=message,
            metadata={
                "rule_name": rule.name,
                "value": value,
                "threshold": rule.threshold,
                "condition": rule.condition
            }
        )
    
    def _process_alert(self, alert: Alert):
        """Procesa una alerta activada"""
        alert_key = f"{alert.type}_{alert.level.value}"
        
        # Agregar al historial
        self.alert_history.append(alert)
        
        # Actualizar alertas activas
        if alert_key not in self.active_alerts:
            self.active_alerts[alert_key] = alert
            logger.warning(f"Alert triggered: {alert.message}")
            
            # Ejecutar callbacks
            for callback in self.callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
        else:
            # Actualizar alerta existente
            self.active_alerts[alert_key] = alert
    
    def resolve_alert(self, alert_key: str):
        """Resuelve una alerta activa"""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_key]
            logger.info(f"Alert resolved: {alert_key}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Obtiene todas las alertas activas"""
        return list(self.active_alerts.values())
    
    def get_alert_history(
        self,
        hours: int = 24,
        level: Optional[AlertLevel] = None
    ) -> List[Alert]:
        """
        Obtiene el historial de alertas.
        
        Args:
            hours: Número de horas de historial
            level: Filtrar por nivel (opcional)
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        history = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff
        ]
        
        if level:
            history = [alert for alert in history if alert.level == level]
        
        return sorted(history, key=lambda x: x.timestamp, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de alertas"""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)
        
        by_level = {}
        for alert in self.alert_history:
            level = alert.level.value
            by_level[level] = by_level.get(level, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_count,
            "alerts_by_level": by_level,
            "rules_count": len(self.rules)
        }


# Instancia global del gestor de alertas
_alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    """Obtiene la instancia global del gestor de alertas"""
    return _alert_manager

