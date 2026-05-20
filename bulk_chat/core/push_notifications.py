"""
Push Notifications - Notificaciones Push
========================================

Sistema de notificaciones push en tiempo real con múltiples canales y prioridades.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Canal de notificación."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SLACK = "slack"
    TEAMS = "teams"


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
    user_id: str
    title: str
    message: str
    channels: List[NotificationChannel]
    priority: NotificationPriority = NotificationPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    action_url: Optional[str] = None


@dataclass
class NotificationSubscription:
    """Suscripción de notificaciones."""
    user_id: str
    channels: List[NotificationChannel]
    preferences: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class PushNotifications:
    """Sistema de notificaciones push."""
    
    def __init__(self):
        self.notifications: Dict[str, Notification] = {}
        self.subscriptions: Dict[str, NotificationSubscription] = {}
        self.notification_queue: deque = deque()
        self.delivery_handlers: Dict[NotificationChannel, Callable] = {}
        self._lock = asyncio.Lock()
        self._processing = False
    
    def register_delivery_handler(
        self,
        channel: NotificationChannel,
        handler: Callable,
    ):
        """Registrar handler de entrega."""
        self.delivery_handlers[channel] = handler
        logger.info(f"Registered delivery handler for {channel.value}")
    
    async def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        channels: Optional[List[NotificationChannel]] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        action_url: Optional[str] = None,
    ) -> str:
        """Enviar notificación."""
        notification_id = f"notif_{user_id}_{datetime.now().timestamp()}"
        
        # Obtener canales del usuario o usar los proporcionados
        subscription = self.subscriptions.get(user_id)
        if channels is None:
            if subscription:
                channels = subscription.channels if subscription.enabled else []
            else:
                channels = [NotificationChannel.IN_APP]
        
        notification = Notification(
            notification_id=notification_id,
            user_id=user_id,
            title=title,
            message=message,
            channels=channels,
            priority=priority,
            metadata=metadata or {},
            action_url=action_url,
        )
        
        async with self._lock:
            self.notifications[notification_id] = notification
            self.notification_queue.append(notification_id)
        
        # Procesar en background
        asyncio.create_task(self._process_notification(notification))
        
        logger.info(f"Notification queued: {notification_id} for user {user_id}")
        return notification_id
    
    async def _process_notification(self, notification: Notification):
        """Procesar notificación."""
        notification.sent_at = datetime.now()
        
        # Enviar por cada canal
        for channel in notification.channels:
            handler = self.delivery_handlers.get(channel)
            if handler:
                try:
                    await handler(notification)
                    logger.debug(f"Notification sent via {channel.value}: {notification.notification_id}")
                except Exception as e:
                    logger.error(f"Error sending notification via {channel.value}: {e}")
            else:
                logger.warning(f"No handler registered for channel {channel.value}")
        
        notification.delivered_at = datetime.now()
    
    def subscribe_user(
        self,
        user_id: str,
        channels: List[NotificationChannel],
        preferences: Optional[Dict[str, Any]] = None,
    ):
        """Suscribir usuario a notificaciones."""
        subscription = NotificationSubscription(
            user_id=user_id,
            channels=channels,
            preferences=preferences or {},
        )
        
        self.subscriptions[user_id] = subscription
        logger.info(f"User {user_id} subscribed to {len(channels)} channels")
    
    def unsubscribe_user(self, user_id: str):
        """Desuscribir usuario."""
        if user_id in self.subscriptions:
            self.subscriptions[user_id].enabled = False
            logger.info(f"User {user_id} unsubscribed")
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Marcar notificación como leída."""
        notification = self.notifications.get(notification_id)
        if not notification:
            return False
        
        notification.read_at = datetime.now()
        return True
    
    def get_user_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Obtener notificaciones de usuario."""
        notifications = [
            n for n in self.notifications.values()
            if n.user_id == user_id
        ]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read_at]
        
        notifications.sort(key=lambda n: n.created_at, reverse=True)
        
        return [
            {
                "notification_id": n.notification_id,
                "title": n.title,
                "message": n.message,
                "priority": n.priority.value,
                "channels": [c.value for c in n.channels],
                "created_at": n.created_at.isoformat(),
                "sent_at": n.sent_at.isoformat() if n.sent_at else None,
                "delivered_at": n.delivered_at.isoformat() if n.delivered_at else None,
                "read_at": n.read_at.isoformat() if n.read_at else None,
                "action_url": n.action_url,
            }
            for n in notifications[:limit]
        ]
    
    def get_notification_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estadísticas de notificaciones."""
        notifications = list(self.notifications.values())
        
        if user_id:
            notifications = [n for n in notifications if n.user_id == user_id]
        
        by_priority: Dict[str, int] = defaultdict(int)
        by_channel: Dict[str, int] = defaultdict(int)
        unread_count = 0
        
        for n in notifications:
            by_priority[n.priority.value] += 1
            for channel in n.channels:
                by_channel[channel.value] += 1
            if not n.read_at:
                unread_count += 1
        
        return {
            "total_notifications": len(notifications),
            "unread_count": unread_count,
            "by_priority": dict(by_priority),
            "by_channel": dict(by_channel),
        }
















