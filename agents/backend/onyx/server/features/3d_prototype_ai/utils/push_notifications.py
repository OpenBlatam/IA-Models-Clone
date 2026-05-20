"""
Push Notifications - Sistema de notificaciones push avanzado
=============================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class NotificationPriority(str, Enum):
    """Prioridades de notificación"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationChannel(str, Enum):
    """Canales de notificación"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


@dataclass
class PushNotification:
    """Notificación push"""
    id: str
    user_id: str
    title: str
    body: str
    priority: NotificationPriority
    channels: List[NotificationChannel]
    data: Dict[str, Any]
    created_at: datetime
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    status: str = "pending"


class PushNotificationSystem:
    """Sistema de notificaciones push avanzado"""
    
    def __init__(self):
        self.notifications: Dict[str, PushNotification] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.delivery_queue: List[PushNotification] = []
    
    def create_notification(self, user_id: str, title: str, body: str,
                           priority: NotificationPriority = NotificationPriority.NORMAL,
                           channels: Optional[List[NotificationChannel]] = None,
                           data: Optional[Dict[str, Any]] = None) -> PushNotification:
        """Crea una notificación"""
        from uuid import uuid4
        
        notification = PushNotification(
            id=str(uuid4()),
            user_id=user_id,
            title=title,
            body=body,
            priority=priority,
            channels=channels or [NotificationChannel.IN_APP],
            data=data or {},
            created_at=datetime.now()
        )
        
        self.notifications[notification.id] = notification
        self.delivery_queue.append(notification)
        
        logger.info(f"Notificación creada: {notification.id} para usuario {user_id}")
        return notification
    
    async def send_notification(self, notification_id: str):
        """Envía una notificación"""
        notification = self.notifications.get(notification_id)
        if not notification:
            raise ValueError(f"Notificación no encontrada: {notification_id}")
        
        # Obtener preferencias del usuario
        preferences = self.user_preferences.get(notification.user_id, {})
        
        # Filtrar canales según preferencias
        allowed_channels = [
            ch for ch in notification.channels
            if preferences.get(f"{ch.value}_enabled", True)
        ]
        
        if not allowed_channels:
            logger.warning(f"No hay canales habilitados para usuario {notification.user_id}")
            notification.status = "failed"
            return
        
        # Enviar por cada canal
        for channel in allowed_channels:
            try:
                await self._send_by_channel(notification, channel)
            except Exception as e:
                logger.error(f"Error enviando notificación por {channel.value}: {e}")
        
        notification.sent_at = datetime.now()
        notification.status = "sent"
    
    async def _send_by_channel(self, notification: PushNotification, channel: NotificationChannel):
        """Envía notificación por canal específico"""
        if channel == NotificationChannel.IN_APP:
            # Notificación in-app (ya está en el sistema)
            notification.delivered_at = datetime.now()
            logger.info(f"Notificación in-app enviada: {notification.id}")
        
        elif channel == NotificationChannel.EMAIL:
            # Enviar email (simulado)
            logger.info(f"Email enviado a {notification.user_id}: {notification.title}")
            notification.delivered_at = datetime.now()
        
        elif channel == NotificationChannel.SMS:
            # Enviar SMS (simulado)
            logger.info(f"SMS enviado a {notification.user_id}: {notification.title}")
            notification.delivered_at = datetime.now()
        
        elif channel == NotificationChannel.PUSH:
            # Push notification (simulado)
            logger.info(f"Push notification enviada a {notification.user_id}: {notification.title}")
            notification.delivered_at = datetime.now()
        
        elif channel == NotificationChannel.WEBHOOK:
            # Webhook (simulado)
            logger.info(f"Webhook enviado para {notification.user_id}: {notification.title}")
            notification.delivered_at = datetime.now()
    
    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Establece preferencias de notificación del usuario"""
        self.user_preferences[user_id] = preferences
        logger.info(f"Preferencias actualizadas para usuario {user_id}")
    
    def mark_as_read(self, notification_id: str, user_id: str):
        """Marca notificación como leída"""
        notification = self.notifications.get(notification_id)
        if notification and notification.user_id == user_id:
            notification.read_at = datetime.now()
            logger.info(f"Notificación {notification_id} marcada como leída")
    
    def get_user_notifications(self, user_id: str, unread_only: bool = False,
                              limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene notificaciones de un usuario"""
        notifications = [
            n for n in self.notifications.values()
            if n.user_id == user_id
        ]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read_at]
        
        notifications.sort(key=lambda x: x.created_at, reverse=True)
        
        return [
            {
                "id": n.id,
                "title": n.title,
                "body": n.body,
                "priority": n.priority.value,
                "created_at": n.created_at.isoformat(),
                "read_at": n.read_at.isoformat() if n.read_at else None,
                "status": n.status,
                "data": n.data
            }
            for n in notifications[:limit]
        ]
    
    async def process_delivery_queue(self):
        """Procesa cola de entrega"""
        while self.delivery_queue:
            notification = self.delivery_queue.pop(0)
            await self.send_notification(notification.id)




