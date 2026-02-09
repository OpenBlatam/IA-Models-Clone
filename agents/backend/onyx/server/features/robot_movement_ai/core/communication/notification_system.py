"""
Notification System
===================

Sistema de notificaciones.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Tipo de notificación."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class NotificationChannel(Enum):
    """Canal de notificación."""
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TELEGRAM = "telegram"


@dataclass
class Notification:
    """Notificación."""
    notification_id: str
    title: str
    message: str
    notification_type: NotificationType
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    read: bool = False


class NotificationSystem:
    """
    Sistema de notificaciones.
    
    Gestiona y envía notificaciones a través de múltiples canales.
    """
    
    def __init__(self):
        """Inicializar sistema de notificaciones."""
        self.notifications: List[Notification] = []
        self.handlers: Dict[NotificationChannel, List[Callable]] = defaultdict(list)
        self.max_notifications = 1000
    
    def register_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[Notification], None]
    ) -> None:
        """
        Registrar handler para canal.
        
        Args:
            channel: Canal de notificación
            handler: Función handler
        """
        self.handlers[channel].append(handler)
        logger.info(f"Registered handler for channel: {channel.value}")
    
    def send_notification(
        self,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        channels: Optional[List[NotificationChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """
        Enviar notificación.
        
        Args:
            title: Título de la notificación
            message: Mensaje
            notification_type: Tipo de notificación
            channels: Canales a usar (None = todos)
            metadata: Metadata adicional
            
        Returns:
            Notificación creada
        """
        notification_id = f"notif_{len(self.notifications)}"
        
        notification = Notification(
            notification_id=notification_id,
            title=title,
            message=message,
            notification_type=notification_type,
            channels=channels or [NotificationChannel.LOG],
            metadata=metadata or {}
        )
        
        # Enviar a handlers
        channels_to_use = channels or [NotificationChannel.LOG]
        for channel in channels_to_use:
            if channel in self.handlers:
                for handler in self.handlers[channel]:
                    try:
                        handler(notification)
                    except Exception as e:
                        logger.error(f"Error in notification handler: {e}")
        
        # Guardar notificación
        self.notifications.append(notification)
        
        # Limitar tamaño
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        logger.info(f"Notification sent: {title} ({notification_type.value})")
        
        return notification
    
    def get_notifications(
        self,
        notification_type: Optional[NotificationType] = None,
        unread_only: bool = False,
        limit: int = 100
    ) -> List[Notification]:
        """
        Obtener notificaciones.
        
        Args:
            notification_type: Filtrar por tipo
            unread_only: Solo no leídas
            limit: Límite de resultados
            
        Returns:
            Lista de notificaciones
        """
        notifications = self.notifications
        
        if notification_type:
            notifications = [n for n in notifications if n.notification_type == notification_type]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        return notifications[-limit:]
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Marcar notificación como leída."""
        for notification in self.notifications:
            if notification.notification_id == notification_id:
                notification.read = True
                return True
        return False
    
    def mark_all_as_read(self) -> int:
        """Marcar todas las notificaciones como leídas."""
        count = 0
        for notification in self.notifications:
            if not notification.read:
                notification.read = True
                count += 1
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de notificaciones."""
        total = len(self.notifications)
        unread = sum(1 for n in self.notifications if not n.read)
        
        by_type = {}
        for ntype in NotificationType:
            by_type[ntype.value] = sum(
                1 for n in self.notifications
                if n.notification_type == ntype
            )
        
        return {
            "total": total,
            "unread": unread,
            "read": total - unread,
            "by_type": by_type
        }


# Instancia global
_notification_system: Optional[NotificationSystem] = None


def get_notification_system() -> NotificationSystem:
    """Obtener instancia global del sistema de notificaciones."""
    global _notification_system
    if _notification_system is None:
        _notification_system = NotificationSystem()
    return _notification_system

