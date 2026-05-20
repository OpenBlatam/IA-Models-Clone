"""
MCP Advanced Monitoring - Monitoreo avanzado
=============================================
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Alert(BaseModel):
    """Alerta del sistema"""
    alert_id: str = Field(..., description="ID único de la alerta")
    level: AlertLevel = Field(..., description="Nivel de la alerta")
    message: str = Field(..., description="Mensaje de la alerta")
    component: str = Field(..., description="Componente que generó la alerta")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertRule(BaseModel):
    """Regla de alerta"""
    rule_id: str = Field(..., description="ID único de la regla")
    name: str = Field(..., description="Nombre de la regla")
    condition: Any = Field(..., description="Función que evalúa la condición (callable)")
    level: AlertLevel = Field(..., description="Nivel de alerta")
    enabled: bool = Field(default=True, description="Regla habilitada")


class MonitoringSystem:
    """
    Sistema de monitoreo avanzado
    
    Monitorea métricas y genera alertas automáticamente.
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._alerts: List[Alert] = []
        self._alert_rules: List[AlertRule] = []
        self._handlers: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
    
    def record_metric(self, metric_name: str, value: float):
        """
        Registra una métrica
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor de la métrica
        """
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        
        self._metrics[metric_name].append(value)
        
        # Mantener solo últimos 1000 valores
        if len(self._metrics[metric_name]) > 1000:
            self._metrics[metric_name] = self._metrics[metric_name][-1000:]
        
        # Evaluar reglas de alerta
        self._evaluate_alert_rules(metric_name, value)
    
    def register_alert_rule(self, rule: AlertRule):
        """
        Registra una regla de alerta
        
        Args:
            rule: Regla de alerta
        """
        self._alert_rules.append(rule)
        logger.info(f"Registered alert rule: {rule.name}")
    
    def register_alert_handler(self, level: AlertLevel, handler: Callable):
        """
        Registra handler para alertas
        
        Args:
            level: Nivel de alerta
            handler: Función handler
        """
        self._handlers[level].append(handler)
        logger.info(f"Registered alert handler for level {level.value}")
    
    def _evaluate_alert_rules(self, metric_name: str, value: float):
        """Evalúa reglas de alerta"""
        for rule in self._alert_rules:
            if not rule.enabled:
                continue
            
            try:
                if rule.condition(metric_name, value):
                    self._trigger_alert(
                        level=rule.level,
                        message=f"Alert rule '{rule.name}' triggered for {metric_name}",
                        component=metric_name,
                    )
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.rule_id}: {e}")
    
    def _trigger_alert(
        self,
        level: AlertLevel,
        message: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Dispara una alerta
        
        Args:
            level: Nivel de alerta
            message: Mensaje de la alerta
            component: Componente
            metadata: Metadata adicional
        """
        import uuid
        
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            level=level,
            message=message,
            component=component,
            metadata=metadata or {},
        )
        
        self._alerts.append(alert)
        
        # Mantener solo últimos 1000 alertas
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-1000:]
        
        # Ejecutar handlers
        handlers = self._handlers[level]
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(alert))
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.warning(f"Alert triggered: {level.value} - {message}")
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Obtiene métricas
        
        Args:
            metric_name: Nombre de la métrica (opcional)
            start_time: Tiempo de inicio (opcional)
            
        Returns:
            Diccionario con métricas
        """
        if metric_name:
            values = self._metrics.get(metric_name, [])
            if values:
                return {
                    metric_name: {
                        "count": len(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "latest": values[-1],
                    }
                }
            return {}
        
        return {
            name: {
                "count": len(vals),
                "avg": sum(vals) / len(vals) if vals else 0,
                "min": min(vals) if vals else 0,
                "max": max(vals) if vals else 0,
                "latest": vals[-1] if vals else 0,
            }
            for name, vals in self._metrics.items()
        }
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        component: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """
        Obtiene alertas
        
        Args:
            level: Filtrar por nivel (opcional)
            component: Filtrar por componente (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de alertas
        """
        alerts = self._alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if component:
            alerts = [a for a in alerts if a.component == component]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

