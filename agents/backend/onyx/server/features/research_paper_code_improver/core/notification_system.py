"""
Notification System - Sistema avanzado de notificaciones
=========================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Canales de notificación"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


class NotificationPriority(Enum):
    """Prioridades de notificación"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(Enum):
    """Estados de notificación"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    READ = "read"


@dataclass
class Notification:
    """Notificación individual"""
    id: str
    user_id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority = NotificationPriority.MEDIUM
    status: NotificationStatus = NotificationStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "message": self.message,
            "channel": self.channel.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "retry_count": self.retry_count
        }


class NotificationSystem:
    """Sistema de notificaciones avanzado"""
    
    def __init__(self):
        self.notifications: Dict[str, Notification] = {}
        self.user_preferences: Dict[str, Dict[NotificationChannel, bool]] = {}
        self.channel_handlers: Dict[NotificationChannel, Callable] = {}
        self.templates: Dict[str, Dict[str, str]] = {}
        self.queue: List[Notification] = []
        self._processing = False
    
    def register_channel_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[Notification], bool]
    ):
        """Registra un handler para un canal"""
        self.channel_handlers[channel] = handler
        logger.info(f"Handler registrado para canal {channel.value}")
    
    def create_template(
        self,
        template_id: str,
        title_template: str,
        message_template: str
    ):
        """Crea una plantilla de notificación"""
        self.templates[template_id] = {
            "title": title_template,
            "message": message_template
        }
    
    def create_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        channel: NotificationChannel,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        template_id: Optional[str] = None,
        template_vars: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Crea una nueva notificación"""
        import uuid
        
        # Usar template si está especificado
        if template_id and template_id in self.templates:
            template = self.templates[template_id]
            title = self._render_template(template["title"], template_vars or {})
            message = self._render_template(template["message"], template_vars or {})
        
        notification = Notification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            message=message,
            channel=channel,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.notifications[notification.id] = notification
        self.queue.append(notification)
        
        # Procesar cola si no está procesando
        if not self._processing:
            asyncio.create_task(self._process_queue())
        
        return notification
    
    def _render_template(self, template: str, vars: Dict[str, Any]) -> str:
        """Renderiza una plantilla"""
        result = template
        for key, value in vars.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result
    
    async def _process_queue(self):
        """Procesa la cola de notificaciones"""
        if self._processing:
            return
        
        self._processing = True
        
        while self.queue:
            notification = self.queue.pop(0)
            
            # Verificar preferencias del usuario
            if not self._should_send(notification):
                notification.status = NotificationStatus.FAILED
                continue
            
            # Enviar notificación
            success = await self._send_notification(notification)
            
            if success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now()
            else:
                notification.retry_count += 1
                if notification.retry_count < notification.max_retries:
                    # Reintentar después de un delay
                    await asyncio.sleep(5 * notification.retry_count)
                    self.queue.append(notification)
                else:
                    notification.status = NotificationStatus.FAILED
        
        self._processing = False
    
    def _should_send(self, notification: Notification) -> bool:
        """Verifica si se debe enviar la notificación"""
        user_prefs = self.user_preferences.get(notification.user_id, {})
        channel_enabled = user_prefs.get(notification.channel, True)
        return channel_enabled
    
    async def _send_notification(self, notification: Notification) -> bool:
        """Envía una notificación"""
        if notification.channel not in self.channel_handlers:
            logger.warning(f"No hay handler para canal {notification.channel.value}")
            return False
        
        handler = self.channel_handlers[notification.channel]
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(notification)
            else:
                result = handler(notification)
            
            return result
        except Exception as e:
            logger.error(f"Error enviando notificación {notification.id}: {e}")
            return False
    
    def set_user_preferences(
        self,
        user_id: str,
        preferences: Dict[NotificationChannel, bool]
    ):
        """Establece preferencias de notificación del usuario"""
        self.user_preferences[user_id] = preferences
    
    def get_user_notifications(
        self,
        user_id: str,
        status: Optional[NotificationStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Obtiene notificaciones de un usuario"""
        notifications = [
            n for n in self.notifications.values()
            if n.user_id == user_id
        ]
        
        if status:
            notifications = [n for n in notifications if n.status == status]
        
        # Ordenar por fecha (más recientes primero)
        notifications.sort(key=lambda n: n.created_at, reverse=True)
        
        return [n.to_dict() for n in notifications[:limit]]
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Marca una notificación como leída"""
        if notification_id not in self.notifications:
            return False
        
        notification = self.notifications[notification_id]
        notification.status = NotificationStatus.READ
        notification.read_at = datetime.now()
        return True
    
    def get_notification_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene estadísticas de notificaciones"""
        notifications = list(self.notifications.values())
        
        if user_id:
            notifications = [n for n in notifications if n.user_id == user_id]
        
        stats = {
            "total": len(notifications),
            "pending": len([n for n in notifications if n.status == NotificationStatus.PENDING]),
            "sent": len([n for n in notifications if n.status == NotificationStatus.SENT]),
            "delivered": len([n for n in notifications if n.status == NotificationStatus.DELIVERED]),
            "read": len([n for n in notifications if n.status == NotificationStatus.READ]),
            "failed": len([n for n in notifications if n.status == NotificationStatus.FAILED]),
            "by_channel": {},
            "by_priority": {}
        }
        
        # Estadísticas por canal
        for channel in NotificationChannel:
            count = len([n for n in notifications if n.channel == channel])
            stats["by_channel"][channel.value] = count
        
        # Estadísticas por prioridad
        for priority in NotificationPriority:
            count = len([n for n in notifications if n.priority == priority])
            stats["by_priority"][priority.value] = count
        
        return stats




