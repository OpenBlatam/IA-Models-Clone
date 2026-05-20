"""
Notifications - Sistema de notificaciones
=========================================

Sistema de notificaciones para alertas y eventos.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Niveles de notificación"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Notification:
    """Notificación"""
    id: str
    title: str
    message: str
    level: NotificationLevel
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    read: bool = False


class NotificationManager:
    """Gestor de notificaciones"""
    
    def __init__(self, max_notifications: int = 1000):
        self.notifications: List[Notification] = []
        self.max_notifications = max_notifications
        self.subscribers: List[Callable] = []
        self._notification_id_counter = 0
    
    def subscribe(self, callback: Callable):
        """Suscribirse a notificaciones"""
        self.subscribers.append(callback)
        logger.info("📬 New notification subscriber")
    
    def unsubscribe(self, callback: Callable):
        """Desuscribirse de notificaciones"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def notify(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Crear y enviar notificación"""
        notification = Notification(
            id=f"notif_{self._notification_id_counter}_{datetime.now().timestamp()}",
            title=title,
            message=message,
            level=level,
            metadata=metadata or {}
        )
        
        self._notification_id_counter += 1
        
        # Agregar a la lista
        self.notifications.append(notification)
        
        # Limitar tamaño
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        # Notificar a suscriptores
        await self._notify_subscribers(notification)
        
        logger.info(f"📢 Notification: {level.value.upper()} - {title}")
        
        return notification
    
    async def _notify_subscribers(self, notification: Notification):
        """Notificar a todos los suscriptores"""
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(notification)
                else:
                    subscriber(notification)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def get_notifications(
        self,
        level: Optional[NotificationLevel] = None,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Notification]:
        """Obtener notificaciones"""
        notifications = self.notifications
        
        # Filtrar por nivel
        if level:
            notifications = [n for n in notifications if n.level == level]
        
        # Filtrar no leídas
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        # Ordenar por timestamp (más recientes primero)
        notifications.sort(key=lambda x: x.timestamp, reverse=True)
        
        return notifications[:limit]
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Marcar notificación como leída"""
        for notification in self.notifications:
            if notification.id == notification_id:
                notification.read = True
                return True
        return False
    
    def mark_all_as_read(self):
        """Marcar todas las notificaciones como leídas"""
        for notification in self.notifications:
            notification.read = True
    
    def delete_notification(self, notification_id: str) -> bool:
        """Eliminar notificación"""
        for i, notification in enumerate(self.notifications):
            if notification.id == notification_id:
                del self.notifications[i]
                return True
        return False
    
    def clear_all(self):
        """Limpiar todas las notificaciones"""
        self.notifications.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de notificaciones"""
        total = len(self.notifications)
        unread = sum(1 for n in self.notifications if not n.read)
        
        by_level = {}
        for level in NotificationLevel:
            by_level[level.value] = sum(1 for n in self.notifications if n.level == level)
        
        return {
            "total": total,
            "unread": unread,
            "read": total - unread,
            "by_level": by_level
        }



