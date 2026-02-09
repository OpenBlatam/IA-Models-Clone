"""
Sistema de Notificaciones Avanzado

Soporta múltiples canales:
- WebSocket (tiempo real)
- Email
- Push notifications
- SMS (futuro)
- Webhooks
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Canales de notificación"""
    WEBSOCKET = "websocket"
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"
    WEBHOOK = "webhook"


class NotificationPriority(Enum):
    """Prioridades de notificación"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Representa una notificación"""
    id: str
    user_id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority = NotificationPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    delivered: bool = False
    error: Optional[str] = None


class AdvancedNotificationService:
    """Servicio avanzado de notificaciones"""
    
    def __init__(self):
        self.notifications: Dict[str, Notification] = {}
        self.channel_handlers: Dict[NotificationChannel, Callable] = {}
        self._setup_default_handlers()
        logger.info("AdvancedNotificationService initialized")
    
    def _setup_default_handlers(self):
        """Configura handlers por defecto"""
        self.channel_handlers[NotificationChannel.WEBSOCKET] = self._send_websocket
        self.channel_handlers[NotificationChannel.EMAIL] = self._send_email
        self.channel_handlers[NotificationChannel.PUSH] = self._send_push
        self.channel_handlers[NotificationChannel.WEBHOOK] = self._send_webhook
    
    async def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        channel: NotificationChannel,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None,
        notification_id: Optional[str] = None
    ) -> str:
        """
        Envía una notificación
        
        Args:
            user_id: ID del usuario
            title: Título de la notificación
            message: Mensaje
            channel: Canal de notificación
            priority: Prioridad
            data: Datos adicionales
            notification_id: ID opcional
        
        Returns:
            ID de la notificación
        """
        import uuid
        notif_id = notification_id or str(uuid.uuid4())
        
        notification = Notification(
            id=notif_id,
            user_id=user_id,
            title=title,
            message=message,
            channel=channel,
            priority=priority,
            data=data or {}
        )
        
        self.notifications[notif_id] = notification
        
        # Enviar a través del canal
        handler = self.channel_handlers.get(channel)
        if handler:
            try:
                await handler(notification)
                notification.delivered = True
                notification.sent_at = datetime.now()
                logger.info(f"Notification {notif_id} sent via {channel.value}")
            except Exception as e:
                notification.error = str(e)
                logger.error(f"Error sending notification {notif_id}: {e}")
        else:
            logger.warning(f"No handler for channel {channel.value}")
        
        return notif_id
    
    async def send_multichannel(
        self,
        user_id: str,
        title: str,
        message: str,
        channels: List[NotificationChannel],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Envía una notificación a múltiples canales"""
        notification_ids = []
        for channel in channels:
            notif_id = await self.send_notification(
                user_id, title, message, channel, priority, data
            )
            notification_ids.append(notif_id)
        return notification_ids
    
    async def _send_websocket(self, notification: Notification):
        """Envía notificación vía WebSocket"""
        try:
            from api.websocket_api import manager
            
            await manager.broadcast_to_user({
                "type": "notification",
                "id": notification.id,
                "title": notification.title,
                "message": notification.message,
                "priority": notification.priority.value,
                "data": notification.data,
                "timestamp": notification.created_at.isoformat()
            }, notification.user_id)
        except Exception as e:
            logger.error(f"Error sending WebSocket notification: {e}")
            raise
    
    async def _send_email(self, notification: Notification):
        """Envía notificación vía email"""
        # Implementación básica - se puede integrar con SendGrid, AWS SES, etc.
        logger.info(f"Email notification to {notification.user_id}: {notification.title}")
        # Aquí iría la lógica real de envío de email
        await asyncio.sleep(0.1)  # Simular envío
    
    async def _send_push(self, notification: Notification):
        """Envía notificación push"""
        # Implementación básica - se puede integrar con FCM, APNS, etc.
        logger.info(f"Push notification to {notification.user_id}: {notification.title}")
        # Aquí iría la lógica real de push
        await asyncio.sleep(0.1)  # Simular envío
    
    async def _send_webhook(self, notification: Notification):
        """Envía notificación vía webhook"""
        webhook_url = notification.data.get("webhook_url")
        if not webhook_url:
            raise ValueError("webhook_url required for webhook notifications")
        
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                webhook_url,
                json={
                    "id": notification.id,
                    "user_id": notification.user_id,
                    "title": notification.title,
                    "message": notification.message,
                    "priority": notification.priority.value,
                    "data": notification.data,
                    "timestamp": notification.created_at.isoformat()
                },
                timeout=10.0
            )
    
    def get_user_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Notification]:
        """Obtiene notificaciones de un usuario"""
        notifications = [
            n for n in self.notifications.values()
            if n.user_id == user_id
        ]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read_at]
        
        return sorted(notifications, key=lambda x: x.created_at, reverse=True)[:limit]
    
    def mark_as_read(self, notification_id: str):
        """Marca una notificación como leída"""
        if notification_id in self.notifications:
            self.notifications[notification_id].read_at = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de notificaciones"""
        total = len(self.notifications)
        by_channel = {}
        by_priority = {}
        delivered = sum(1 for n in self.notifications.values() if n.delivered)
        read = sum(1 for n in self.notifications.values() if n.read_at)
        
        for notification in self.notifications.values():
            channel = notification.channel.value
            priority = notification.priority.value
            by_channel[channel] = by_channel.get(channel, 0) + 1
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        return {
            "total": total,
            "delivered": delivered,
            "read": read,
            "unread": total - read,
            "by_channel": by_channel,
            "by_priority": by_priority,
            "delivery_rate": (delivered / total * 100) if total > 0 else 0
        }


# Instancia global
_notification_service: Optional[AdvancedNotificationService] = None


def get_notification_service() -> AdvancedNotificationService:
    """Obtiene la instancia global del servicio de notificaciones"""
    global _notification_service
    if _notification_service is None:
        _notification_service = AdvancedNotificationService()
    return _notification_service

