"""
Notifications - Sistema de notificaciones
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Tipos de notificaciones"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    OPERATION_COMPLETE = "operation_complete"
    BACKUP_CREATED = "backup_created"
    VERSION_CREATED = "version_created"


class Notification:
    """Clase para representar una notificación"""

    def __init__(
        self,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar notificación.

        Args:
            title: Título de la notificación
            message: Mensaje
            notification_type: Tipo de notificación
            metadata: Metadatos adicionales
        """
        self.id = f"notif_{datetime.utcnow().timestamp()}"
        self.title = title
        self.message = message
        self.type = notification_type
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()
        self.read = False


class NotificationManager:
    """Gestor de notificaciones"""

    def __init__(self):
        """Inicializar el gestor de notificaciones"""
        self.notifications: List[Notification] = []
        self.subscribers: List[Callable] = []
        self.max_notifications = 1000

    def notify(
        self,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """
        Crear y enviar una notificación.

        Args:
            title: Título
            message: Mensaje
            notification_type: Tipo
            metadata: Metadatos

        Returns:
            Notificación creada
        """
        notification = Notification(title, message, notification_type, metadata)
        
        # Agregar a la lista
        self.notifications.append(notification)
        
        # Limitar tamaño
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        # Notificar a suscriptores
        self._notify_subscribers(notification)
        
        logger.info(f"Notificación: {title} - {message}")
        return notification

    def subscribe(self, callback: Callable):
        """
        Suscribirse a notificaciones.

        Args:
            callback: Función callback
        """
        self.subscribers.append(callback)

    def _notify_subscribers(self, notification: Notification):
        """Notificar a todos los suscriptores"""
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(notification))
                else:
                    callback(notification)
            except Exception as e:
                logger.error(f"Error en callback de notificación: {e}")

    def get_notifications(
        self,
        notification_type: Optional[NotificationType] = None,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
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
        
        # Filtrar por tipo
        if notification_type:
            notifications = [n for n in notifications if n.type == notification_type]
        
        # Filtrar por leídas
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        # Limitar
        notifications = notifications[-limit:][::-1]
        
        return [self._notification_to_dict(n) for n in notifications]

    def mark_as_read(self, notification_id: str):
        """
        Marcar notificación como leída.

        Args:
            notification_id: ID de la notificación
        """
        for notification in self.notifications:
            if notification.id == notification_id:
                notification.read = True
                break

    def mark_all_as_read(self):
        """Marcar todas las notificaciones como leídas"""
        for notification in self.notifications:
            notification.read = True

    def _notification_to_dict(self, notification: Notification) -> Dict[str, Any]:
        """Convertir notificación a diccionario"""
        return {
            "id": notification.id,
            "title": notification.title,
            "message": notification.message,
            "type": notification.type.value,
            "metadata": notification.metadata,
            "timestamp": notification.timestamp,
            "read": notification.read
        }

    def clear(self):
        """Limpiar todas las notificaciones"""
        self.notifications.clear()






