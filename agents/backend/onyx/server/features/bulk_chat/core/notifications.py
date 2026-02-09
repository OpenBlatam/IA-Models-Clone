"""
Notifications - Sistema de Notificaciones Push
==============================================

Sistema de notificaciones push en tiempo real.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Tipos de notificación."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SYSTEM = "system"


class NotificationPriority(Enum):
    """Prioridad de notificación."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notificación."""
    notification_id: str
    user_id: str
    title: str
    message: str
    notification_type: NotificationType
    priority: NotificationPriority = NotificationPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    read_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationManager:
    """Gestor de notificaciones."""
    
    def __init__(self):
        self.notifications: Dict[str, List[Notification]] = {}
        self.subscribers: Dict[str, List[Callable]] = {}  # {user_id: [callbacks]}
        self._lock = asyncio.Lock()
    
    async def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Notification:
        """
        Enviar notificación.
        
        Args:
            user_id: ID del usuario
            title: Título de la notificación
            message: Mensaje
            notification_type: Tipo de notificación
            priority: Prioridad
            metadata: Metadatos adicionales
        
        Returns:
            Notificación creada
        """
        notification_id = f"notif_{datetime.now().timestamp()}_{user_id}"
        
        notification = Notification(
            notification_id=notification_id,
            user_id=user_id,
            title=title,
            message=message,
            notification_type=notification_type,
            priority=priority,
            metadata=metadata or {},
        )
        
        async with self._lock:
            if user_id not in self.notifications:
                self.notifications[user_id] = []
            self.notifications[user_id].append(notification)
        
        # Notificar a suscriptores
        await self._notify_subscribers(user_id, notification)
        
        logger.info(f"Sent notification to {user_id}: {title}")
        
        return notification
    
    async def _notify_subscribers(
        self,
        user_id: str,
        notification: Notification,
    ):
        """Notificar a suscriptores."""
        subscribers = self.subscribers.get(user_id, [])
        
        if subscribers:
            tasks = [
                self._call_subscriber(callback, notification)
                for callback in subscribers
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_subscriber(
        self,
        callback: Callable,
        notification: Notification,
    ):
        """Llamar suscriptor."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(notification)
            else:
                callback(notification)
        except Exception as e:
            logger.error(f"Error calling notification subscriber: {e}")
    
    async def subscribe(
        self,
        user_id: str,
        callback: Callable,
    ):
        """Suscribirse a notificaciones de usuario."""
        async with self._lock:
            if user_id not in self.subscribers:
                self.subscribers[user_id] = []
            self.subscribers[user_id].append(callback)
            logger.debug(f"Subscribed to notifications for {user_id}")
    
    async def unsubscribe(
        self,
        user_id: str,
        callback: Callable,
    ):
        """Desuscribirse de notificaciones."""
        async with self._lock:
            if user_id in self.subscribers:
                if callback in self.subscribers[user_id]:
                    self.subscribers[user_id].remove(callback)
                    logger.debug(f"Unsubscribed from notifications for {user_id}")
    
    async def get_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50,
    ) -> List[Notification]:
        """Obtener notificaciones de usuario."""
        notifications = self.notifications.get(user_id, [])
        
        if unread_only:
            notifications = [n for n in notifications if n.read_at is None]
        
        # Ordenar por fecha (más recientes primero)
        notifications.sort(key=lambda n: n.created_at, reverse=True)
        
        return notifications[:limit]
    
    async def mark_as_read(
        self,
        user_id: str,
        notification_id: str,
    ):
        """Marcar notificación como leída."""
        notifications = self.notifications.get(user_id, [])
        
        for notification in notifications:
            if notification.notification_id == notification_id:
                notification.read_at = datetime.now()
                logger.debug(f"Marked notification {notification_id} as read")
                return
        
        logger.warning(f"Notification {notification_id} not found for user {user_id}")
    
    async def mark_all_as_read(self, user_id: str):
        """Marcar todas las notificaciones como leídas."""
        notifications = self.notifications.get(user_id, [])
        
        for notification in notifications:
            if notification.read_at is None:
                notification.read_at = datetime.now()
        
        logger.info(f"Marked all notifications as read for {user_id}")
    
    def get_unread_count(self, user_id: str) -> int:
        """Obtener conteo de notificaciones no leídas."""
        notifications = self.notifications.get(user_id, [])
        return sum(1 for n in notifications if n.read_at is None)
































