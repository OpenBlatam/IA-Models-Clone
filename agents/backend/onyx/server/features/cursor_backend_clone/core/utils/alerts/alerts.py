"""
Alerts - Sistema de Alertas
============================

Sistema de alertas basado en métricas y condiciones.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

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
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    GREATER_OR_EQUAL = "gte"
    LESS_OR_EQUAL = "lte"


@dataclass
class Alert:
    """Alerta"""
    name: str
    message: str
    severity: AlertSeverity
    condition: AlertCondition
    threshold: float
    metric_name: str
    current_value: float
    triggered_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_resolved(self) -> bool:
        """Verificar si la alerta está resuelta"""
        return self.resolved_at is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "name": self.name,
            "message": self.message,
            "severity": self.severity.value,
            "condition": self.condition.value,
            "threshold": self.threshold,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata
        }


class AlertManager:
    """
    Gestor de alertas.
    
    Monitorea métricas y dispara alertas basadas en condiciones.
    """
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.handlers: List[Callable[[Alert], None]] = []
        self.check_interval = 60.0  # segundos
        self._check_task: Optional[asyncio.Task] = None
    
    def add_rule(
        self,
        name: str,
        metric_name: str,
        condition: AlertCondition,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Agregar regla de alerta.
        
        Args:
            name: Nombre de la regla
            metric_name: Nombre de la métrica a monitorear
            condition: Condición a evaluar
            threshold: Valor umbral
            severity: Severidad de la alerta
            message: Mensaje personalizado
            metadata: Metadata adicional
        """
        rule = {
            "name": name,
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "message": message or f"{metric_name} {condition.value} {threshold}",
            "metadata": metadata or {}
        }
        
        self.alert_rules.append(rule)
        logger.info(f"🚨 Alert rule added: {name}")
    
    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        Registrar handler para alertas.
        
        Args:
            handler: Función que maneja alertas (puede ser async o sync)
        """
        self.handlers.append(handler)
        logger.debug("🚨 Alert handler registered")
    
    async def check_metrics(self, metrics_getter: Callable[[str], Optional[float]]) -> None:
        """
        Verificar métricas contra reglas.
        
        Args:
            metrics_getter: Función que obtiene valor de métrica por nombre
        """
        for rule in self.alert_rules:
            metric_value = metrics_getter(rule["metric_name"])
            
            if metric_value is None:
                continue
            
            # Evaluar condición
            should_alert = False
            condition = rule["condition"]
            threshold = rule["threshold"]
            
            if condition == AlertCondition.GREATER_THAN:
                should_alert = metric_value > threshold
            elif condition == AlertCondition.LESS_THAN:
                should_alert = metric_value < threshold
            elif condition == AlertCondition.EQUAL:
                should_alert = metric_value == threshold
            elif condition == AlertCondition.NOT_EQUAL:
                should_alert = metric_value != threshold
            elif condition == AlertCondition.GREATER_OR_EQUAL:
                should_alert = metric_value >= threshold
            elif condition == AlertCondition.LESS_OR_EQUAL:
                should_alert = metric_value <= threshold
            
            # Crear o resolver alerta
            alert_name = rule["name"]
            
            if should_alert:
                # Crear nueva alerta o actualizar existente
                if alert_name not in self.alerts or self.alerts[alert_name].is_resolved():
                    alert = Alert(
                        name=alert_name,
                        message=rule["message"],
                        severity=rule["severity"],
                        condition=condition,
                        threshold=threshold,
                        metric_name=rule["metric_name"],
                        current_value=metric_value,
                        metadata=rule["metadata"]
                    )
                    
                    self.alerts[alert_name] = alert
                    await self._trigger_alert(alert)
            else:
                # Resolver alerta si existe
                if alert_name in self.alerts and not self.alerts[alert_name].is_resolved():
                    self.alerts[alert_name].resolved_at = datetime.now()
                    logger.info(f"✅ Alert resolved: {alert_name}")
    
    async def _trigger_alert(self, alert: Alert) -> None:
        """Disparar alerta a todos los handlers"""
        logger.warning(
            f"🚨 Alert triggered: {alert.name} - {alert.message} "
            f"({alert.metric_name}={alert.current_value})"
        )
        
        for handler in self.handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    async def start_monitoring(
        self,
        metrics_getter: Callable[[str], Optional[float]]
    ) -> None:
        """
        Iniciar monitoreo continuo.
        
        Args:
            metrics_getter: Función que obtiene valor de métrica
        """
        if self._check_task and not self._check_task.done():
            return
        
        self._check_task = asyncio.create_task(
            self._monitoring_loop(metrics_getter)
        )
        logger.info("🚨 Alert monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Detener monitoreo"""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            logger.info("🚨 Alert monitoring stopped")
    
    async def _monitoring_loop(
        self,
        metrics_getter: Callable[[str], Optional[float]]
    ) -> None:
        """Loop de monitoreo"""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self.check_metrics(metrics_getter)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Obtener alertas activas"""
        return [
            alert for alert in self.alerts.values()
            if not alert.is_resolved()
        ]
    
    def get_all_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Obtener todas las alertas con filtros.
        
        Args:
            severity: Filtrar por severidad
            since: Filtrar desde fecha
            limit: Límite de resultados
            
        Returns:
            Lista de alertas
        """
        alerts = list(self.alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since:
            alerts = [a for a in alerts if a.triggered_at >= since]
        
        # Ordenar por severidad y timestamp
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }
        
        alerts.sort(
            key=lambda x: (severity_order.get(x.severity, 99), x.triggered_at),
            reverse=True
        )
        
        return alerts[:limit]




