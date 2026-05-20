"""
Notification Service
====================

Servicio de notificaciones y recordatorios.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Tipos de notificaciones."""
    EVENT_REMINDER = "event_reminder"
    ROUTINE_REMINDER = "routine_reminder"
    PROTOCOL_ALERT = "protocol_alert"
    WARDROBE_REMINDER = "wardrobe_reminder"
    DAILY_SUMMARY = "daily_summary"
    COMPLIANCE_WARNING = "compliance_warning"
    GENERAL = "general"


class NotificationPriority(Enum):
    """Prioridad de notificaciones."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notificación."""
    id: str
    artist_id: str
    type: NotificationType
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    data: Optional[Dict[str, Any]] = None
    scheduled_time: Optional[datetime] = None
    sent: bool = False
    read: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data['type'] = self.type.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        if self.scheduled_time:
            data['scheduled_time'] = self.scheduled_time.isoformat()
        return data


class NotificationService:
    """Servicio de notificaciones."""
    
    def __init__(self):
        """Inicializar servicio de notificaciones."""
        self.notifications: Dict[str, List[Notification]] = {}  # artist_id -> [notifications]
        self._logger = logger
    
    def create_notification(
        self,
        artist_id: str,
        notification_type: NotificationType,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None,
        scheduled_time: Optional[datetime] = None
    ) -> Notification:
        """
        Crear notificación.
        
        Args:
            artist_id: ID del artista
            notification_type: Tipo de notificación
            title: Título
            message: Mensaje
            priority: Prioridad
            data: Datos adicionales
            scheduled_time: Hora programada (opcional)
        
        Returns:
            Notificación creada
        """
        import uuid
        notification = Notification(
            id=str(uuid.uuid4()),
            artist_id=artist_id,
            type=notification_type,
            title=title,
            message=message,
            priority=priority,
            data=data,
            scheduled_time=scheduled_time
        )
        
        if artist_id not in self.notifications:
            self.notifications[artist_id] = []
        
        self.notifications[artist_id].append(notification)
        self._logger.info(f"Notification created for artist {artist_id}: {title}")
        return notification
    
    def get_notifications(
        self,
        artist_id: str,
        unread_only: bool = False,
        notification_type: Optional[NotificationType] = None
    ) -> List[Notification]:
        """
        Obtener notificaciones de un artista.
        
        Args:
            artist_id: ID del artista
            unread_only: Solo no leídas
            notification_type: Filtrar por tipo
        
        Returns:
            Lista de notificaciones
        """
        if artist_id not in self.notifications:
            return []
        
        notifications = self.notifications[artist_id]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        if notification_type:
            notifications = [n for n in notifications if n.type == notification_type]
        
        return sorted(notifications, key=lambda n: n.created_at, reverse=True)
    
    def mark_as_read(self, artist_id: str, notification_id: str) -> bool:
        """
        Marcar notificación como leída.
        
        Args:
            artist_id: ID del artista
            notification_id: ID de la notificación
        
        Returns:
            True si se marcó, False si no se encontró
        """
        if artist_id not in self.notifications:
            return False
        
        for notification in self.notifications[artist_id]:
            if notification.id == notification_id:
                notification.read = True
                return True
        
        return False
    
    def create_event_reminder(
        self,
        artist_id: str,
        event_title: str,
        event_time: datetime,
        minutes_before: int = 60
    ) -> Notification:
        """
        Crear recordatorio de evento.
        
        Args:
            artist_id: ID del artista
            event_title: Título del evento
            event_time: Hora del evento
            minutes_before: Minutos antes del evento
        
        Returns:
            Notificación creada
        """
        reminder_time = event_time - timedelta(minutes=minutes_before)
        
        return self.create_notification(
            artist_id=artist_id,
            notification_type=NotificationType.EVENT_REMINDER,
            title=f"Recordatorio: {event_title}",
            message=f"Tienes un evento en {minutes_before} minutos: {event_title}",
            priority=NotificationPriority.HIGH,
            data={"event_time": event_time.isoformat()},
            scheduled_time=reminder_time
        )
    
    def create_routine_reminder(
        self,
        artist_id: str,
        routine_title: str,
        scheduled_time: datetime
    ) -> Notification:
        """
        Crear recordatorio de rutina.
        
        Args:
            artist_id: ID del artista
            routine_title: Título de la rutina
            scheduled_time: Hora programada
        
        Returns:
            Notificación creada
        """
        return self.create_notification(
            artist_id=artist_id,
            notification_type=NotificationType.ROUTINE_REMINDER,
            title=f"Rutina: {routine_title}",
            message=f"Es hora de realizar tu rutina: {routine_title}",
            priority=NotificationPriority.NORMAL,
            data={"scheduled_time": scheduled_time.isoformat()},
            scheduled_time=scheduled_time
        )
    
    def create_protocol_alert(
        self,
        artist_id: str,
        protocol_title: str,
        event_title: str,
        message: str
    ) -> Notification:
        """
        Crear alerta de protocolo.
        
        Args:
            artist_id: ID del artista
            protocol_title: Título del protocolo
            event_title: Título del evento relacionado
            message: Mensaje de alerta
        
        Returns:
            Notificación creada
        """
        return self.create_notification(
            artist_id=artist_id,
            notification_type=NotificationType.PROTOCOL_ALERT,
            title=f"Alerta de Protocolo: {protocol_title}",
            message=message,
            priority=NotificationPriority.HIGH,
            data={"protocol": protocol_title, "event": event_title}
        )
    
    def get_due_notifications(self, artist_id: Optional[str] = None) -> List[Notification]:
        """
        Obtener notificaciones programadas que deben enviarse.
        
        Args:
            artist_id: ID del artista (opcional)
        
        Returns:
            Lista de notificaciones vencidas
        """
        now = datetime.now()
        due_notifications = []
        
        artists_to_check = [artist_id] if artist_id else self.notifications.keys()
        
        for aid in artists_to_check:
            if aid in self.notifications:
                for notification in self.notifications[aid]:
                    if (notification.scheduled_time and 
                        notification.scheduled_time <= now and 
                        not notification.sent):
                        due_notifications.append(notification)
        
        return due_notifications
    
    def mark_as_sent(self, artist_id: str, notification_id: str) -> bool:
        """
        Marcar notificación como enviada.
        
        Args:
            artist_id: ID del artista
            notification_id: ID de la notificación
        
        Returns:
            True si se marcó, False si no se encontró
        """
        if artist_id not in self.notifications:
            return False
        
        for notification in self.notifications[artist_id]:
            if notification.id == notification_id:
                notification.sent = True
                return True
        
        return False
    
    def delete_notification(self, artist_id: str, notification_id: str) -> bool:
        """
        Eliminar notificación.
        
        Args:
            artist_id: ID del artista
            notification_id: ID de la notificación
        
        Returns:
            True si se eliminó, False si no se encontró
        """
        if artist_id not in self.notifications:
            return False
        
        self.notifications[artist_id] = [
            n for n in self.notifications[artist_id]
            if n.id != notification_id
        ]
        return True




