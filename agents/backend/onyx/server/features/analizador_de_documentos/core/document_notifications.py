"""
Document Notifications - Sistema de Notificaciones
====================================================

Sistema de notificaciones para eventos de análisis.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Canal de notificación."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


class NotificationPriority(Enum):
    """Prioridad de notificación."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notificación."""
    notification_id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority = NotificationPriority.NORMAL
    recipient: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    sent: bool = False
    error: Optional[str] = None


class NotificationManager:
    """Gestor de notificaciones."""
    
    def __init__(self, analyzer):
        """Inicializar gestor."""
        self.analyzer = analyzer
        self.notifications: List[Notification] = []
        self.handlers: Dict[NotificationChannel, List[Callable]] = defaultdict(list)
        self.max_notifications = 5000
    
    def register_handler(
        self,
        channel: NotificationChannel,
        handler: Callable
    ):
        """Registrar handler para canal."""
        self.handlers[channel].append(handler)
        logger.info(f"Handler registrado para canal: {channel.value}")
    
    async def send_notification(
        self,
        title: str,
        message: str,
        channel: NotificationChannel,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        recipient: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """
        Enviar notificación.
        
        Args:
            title: Título de la notificación
            message: Mensaje
            channel: Canal de notificación
            priority: Prioridad
            recipient: Recipiente
            data: Datos adicionales
        
        Returns:
            Notification creada
        """
        notification = Notification(
            notification_id=f"notif_{len(self.notifications) + 1}",
            title=title,
            message=message,
            channel=channel,
            priority=priority,
            recipient=recipient,
            data=data or {}
        )
        
        self.notifications.append(notification)
        
        # Enviar a handlers
        if channel in self.handlers:
            for handler in self.handlers[channel]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(notification)
                    else:
                        handler(notification)
                    notification.sent = True
                except Exception as e:
                    logger.error(f"Error enviando notificación: {e}")
                    notification.error = str(e)
        
        # Mantener solo últimos N
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        return notification
    
    async def notify_analysis_complete(
        self,
        document_id: str,
        result: Any,
        recipient: Optional[str] = None
    ):
        """Notificar análisis completado."""
        await self.send_notification(
            title="Análisis Completado",
            message=f"Análisis del documento {document_id} completado exitosamente",
            channel=NotificationChannel.IN_APP,
            priority=NotificationPriority.NORMAL,
            recipient=recipient,
            data={
                "document_id": document_id,
                "result": result.__dict__ if hasattr(result, '__dict__') else str(result)
            }
        )
    
    async def notify_quality_alert(
        self,
        document_id: str,
        quality_score: float,
        recipient: Optional[str] = None
    ):
        """Notificar alerta de calidad."""
        priority = NotificationPriority.HIGH if quality_score < 50 else NotificationPriority.NORMAL
        
        await self.send_notification(
            title="Alerta de Calidad",
            message=f"Documento {document_id} tiene calidad baja: {quality_score:.1f}/100",
            channel=NotificationChannel.EMAIL,
            priority=priority,
            recipient=recipient,
            data={
                "document_id": document_id,
                "quality_score": quality_score
            }
        )
    
    def get_notifications(
        self,
        channel: Optional[NotificationChannel] = None,
        recipient: Optional[str] = None,
        sent_only: bool = False,
        limit: int = 100
    ) -> List[Notification]:
        """Obtener notificaciones."""
        notifications = self.notifications
        
        if channel:
            notifications = [n for n in notifications if n.channel == channel]
        
        if recipient:
            notifications = [n for n in notifications if n.recipient == recipient]
        
        if sent_only:
            notifications = [n for n in notifications if n.sent]
        
        return notifications[-limit:]
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de notificaciones."""
        total = len(self.notifications)
        by_channel = {
            channel.value: len([n for n in self.notifications if n.channel == channel])
            for channel in NotificationChannel
        }
        by_priority = {
            priority.value: len([n for n in self.notifications if n.priority == priority])
            for priority in NotificationPriority
        }
        sent_count = len([n for n in self.notifications if n.sent])
        error_count = len([n for n in self.notifications if n.error])
        
        return {
            "total_notifications": total,
            "sent": sent_count,
            "errors": error_count,
            "by_channel": by_channel,
            "by_priority": by_priority
        }


__all__ = [
    "NotificationManager",
    "Notification",
    "NotificationChannel",
    "NotificationPriority"
]
















