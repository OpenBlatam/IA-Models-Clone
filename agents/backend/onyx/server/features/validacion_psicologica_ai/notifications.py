"""
Sistema de Notificaciones en Tiempo Real
=========================================
Notificaciones push y en tiempo real para usuarios
"""

from typing import Dict, Any, List, Optional, Callable
from uuid import UUID
from datetime import datetime
from enum import Enum
import structlog
import asyncio
from collections import defaultdict

from .models import ValidationStatus, PsychologicalValidation

logger = structlog.get_logger()


class NotificationType(str, Enum):
    """Tipos de notificación"""
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"
    ALERT_CREATED = "alert_created"
    RECOMMENDATION_AVAILABLE = "recommendation_available"
    PROFILE_UPDATED = "profile_updated"
    REPORT_READY = "report_ready"
    CONNECTION_EXPIRED = "connection_expired"
    SYSTEM_UPDATE = "system_update"


class NotificationPriority(str, Enum):
    """Prioridad de notificación"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Notification:
    """Representa una notificación"""
    
    def __init__(
        self,
        user_id: UUID,
        notification_type: NotificationType,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        data: Optional[Dict[str, Any]] = None,
        read: bool = False
    ):
        self.id = UUID()
        self.user_id = user_id
        self.notification_type = notification_type
        self.title = title
        self.message = message
        self.priority = priority
        self.data = data or {}
        self.read = read
        self.created_at = datetime.utcnow()
        self.read_at: Optional[datetime] = None
    
    def mark_as_read(self) -> None:
        """Marcar notificación como leída"""
        self.read = True
        self.read_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "type": self.notification_type.value,
            "title": self.title,
            "message": self.message,
            "priority": self.priority.value,
            "data": self.data,
            "read": self.read,
            "created_at": self.created_at.isoformat(),
            "read_at": self.read_at.isoformat() if self.read_at else None
        }


class NotificationManager:
    """Gestor de notificaciones"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._notifications: Dict[UUID, List[Notification]] = defaultdict(list)
        self._subscribers: Dict[UUID, List[Callable]] = defaultdict(list)
        logger.info("NotificationManager initialized")
    
    def subscribe(
        self,
        user_id: UUID,
        callback: Callable[[Notification], None]
    ) -> None:
        """
        Suscribirse a notificaciones de un usuario
        
        Args:
            user_id: ID del usuario
            callback: Función callback
        """
        self._subscribers[user_id].append(callback)
        logger.info("Subscriber added", user_id=str(user_id))
    
    def unsubscribe(
        self,
        user_id: UUID,
        callback: Callable[[Notification], None]
    ) -> None:
        """
        Desuscribirse de notificaciones
        
        Args:
            user_id: ID del usuario
            callback: Función callback a remover
        """
        if callback in self._subscribers[user_id]:
            self._subscribers[user_id].remove(callback)
            logger.info("Subscriber removed", user_id=str(user_id))
    
    async def send_notification(
        self,
        user_id: UUID,
        notification_type: NotificationType,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        data: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """
        Enviar notificación
        
        Args:
            user_id: ID del usuario
            notification_type: Tipo de notificación
            title: Título
            message: Mensaje
            priority: Prioridad
            data: Datos adicionales
            
        Returns:
            Notificación creada
        """
        notification = Notification(
            user_id=user_id,
            notification_type=notification_type,
            title=title,
            message=message,
            priority=priority,
            data=data
        )
        
        self._notifications[user_id].append(notification)
        
        # Notificar a suscriptores
        for callback in self._subscribers[user_id]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification)
                else:
                    callback(notification)
            except Exception as e:
                logger.error(
                    "Error in notification callback",
                    error=str(e),
                    user_id=str(user_id)
                )
        
        logger.info(
            "Notification sent",
            notification_id=str(notification.id),
            user_id=str(user_id),
            type=notification_type.value
        )
        
        return notification
    
    def get_notifications(
        self,
        user_id: UUID,
        unread_only: bool = False,
        limit: int = 100
    ) -> List[Notification]:
        """
        Obtener notificaciones de un usuario
        
        Args:
            user_id: ID del usuario
            unread_only: Solo no leídas
            limit: Límite de resultados
            
        Returns:
            Lista de notificaciones
        """
        notifications = self._notifications.get(user_id, [])
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        notifications.sort(key=lambda x: x.created_at, reverse=True)
        return notifications[:limit]
    
    def mark_as_read(
        self,
        user_id: UUID,
        notification_id: UUID
    ) -> bool:
        """
        Marcar notificación como leída
        
        Args:
            user_id: ID del usuario
            notification_id: ID de la notificación
            
        Returns:
            True si se marcó exitosamente
        """
        notifications = self._notifications.get(user_id, [])
        notification = next(
            (n for n in notifications if n.id == notification_id),
            None
        )
        
        if notification:
            notification.mark_as_read()
            return True
        
        return False
    
    def mark_all_as_read(self, user_id: UUID) -> int:
        """
        Marcar todas las notificaciones como leídas
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Número de notificaciones marcadas
        """
        notifications = self._notifications.get(user_id, [])
        unread = [n for n in notifications if not n.read]
        
        for notification in unread:
            notification.mark_as_read()
        
        return len(unread)
    
    def get_unread_count(self, user_id: UUID) -> int:
        """
        Obtener conteo de no leídas
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Número de notificaciones no leídas
        """
        notifications = self._notifications.get(user_id, [])
        return len([n for n in notifications if not n.read])
    
    def delete_notification(
        self,
        user_id: UUID,
        notification_id: UUID
    ) -> bool:
        """
        Eliminar notificación
        
        Args:
            user_id: ID del usuario
            notification_id: ID de la notificación
            
        Returns:
            True si se eliminó exitosamente
        """
        notifications = self._notifications.get(user_id, [])
        notification = next(
            (n for n in notifications if n.id == notification_id),
            None
        )
        
        if notification:
            notifications.remove(notification)
            return True
        
        return False


# Instancia global del gestor de notificaciones
notification_manager = NotificationManager()




