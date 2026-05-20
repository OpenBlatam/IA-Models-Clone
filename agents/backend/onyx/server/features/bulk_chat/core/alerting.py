"""
Alerting System - Sistema de Alertas
=====================================

Sistema avanzado de alertas y notificaciones.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Nivel de alerta."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Tipo de alerta."""
    PERFORMANCE = "performance"
    ERROR = "error"
    RESOURCE = "resource"
    SECURITY = "security"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Alerta."""
    id: str
    type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class AlertManager:
    """Gestor de alertas."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.handlers: Dict[AlertType, List[Callable]] = {}
        self.rules: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def create_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Crear nueva alerta."""
        import uuid
        alert = Alert(
            id=str(uuid.uuid4()),
            type=alert_type,
            level=level,
            title=title,
            message=message,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.alerts[alert.id] = alert
        
        # Notificar handlers
        await self._notify_handlers(alert)
        
        logger.warning(f"Alert created: {alert.title} ({alert.level.value})")
        return alert
    
    async def _notify_handlers(self, alert: Alert):
        """Notificar handlers de alerta."""
        handlers = self.handlers.get(alert.type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def register_handler(self, alert_type: AlertType, handler: Callable):
        """Registrar handler para tipo de alerta."""
        if alert_type not in self.handlers:
            self.handlers[alert_type] = []
        self.handlers[alert_type].append(handler)
        logger.info(f"Registered handler for {alert_type.value}")
    
    async def resolve_alert(self, alert_id: str):
        """Resolver alerta."""
        async with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
    
    def get_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        level: Optional[AlertLevel] = None,
        resolved: Optional[bool] = None,
    ) -> List[Alert]:
        """Obtener alertas."""
        alerts = list(self.alerts.values())
        
        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    async def add_rule(
        self,
        name: str,
        condition: Callable,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message_template: str,
    ):
        """Agregar regla de alerta."""
        rule = {
            "name": name,
            "condition": condition,
            "alert_type": alert_type,
            "level": level,
            "title": title,
            "message_template": message_template,
        }
        
        self.rules.append(rule)
        logger.info(f"Added alert rule: {name}")
    
    async def check_rules(self, context: Dict[str, Any]):
        """Verificar reglas de alerta."""
        for rule in self.rules:
            try:
                if asyncio.iscoroutinefunction(rule["condition"]):
                    triggered = await rule["condition"](context)
                else:
                    triggered = rule["condition"](context)
                
                if triggered:
                    message = rule["message_template"].format(**context)
                    await self.create_alert(
                        rule["alert_type"],
                        rule["level"],
                        rule["title"],
                        message,
                        context,
                    )
            except Exception as e:
                logger.error(f"Error checking rule {rule['name']}: {e}")


class AlertHandler:
    """Handler base para alertas."""
    
    async def handle(self, alert: Alert):
        """Manejar alerta."""
        raise NotImplementedError


class EmailAlertHandler(AlertHandler):
    """Handler para enviar alertas por email."""
    
    def __init__(self, recipients: List[str]):
        self.recipients = recipients
    
    async def handle(self, alert: Alert):
        """Enviar alerta por email."""
        # En producción, implementar envío real de email
        logger.info(f"Email alert sent to {self.recipients}: {alert.title}")


class SlackAlertHandler(AlertHandler):
    """Handler para enviar alertas a Slack."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def handle(self, alert: Alert):
        """Enviar alerta a Slack."""
        # En producción, implementar envío real a Slack
        logger.info(f"Slack alert sent: {alert.title}")
































