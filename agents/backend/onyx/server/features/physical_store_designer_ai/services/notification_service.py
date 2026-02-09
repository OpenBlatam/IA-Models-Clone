"""
Notification Service - Sistema de notificaciones
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Tipos de notificación"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    REMINDER = "reminder"


class NotificationService:
    """Servicio para notificaciones"""
    
    def __init__(self):
        self.notifications: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        title: str,
        message: str,
        action_url: Optional[str] = None,
        priority: str = "normal"  # "low", "normal", "high", "urgent"
    ) -> Dict[str, Any]:
        """Crear notificación"""
        
        notification = {
            "id": f"notif_{user_id}_{len(self.notifications.get(user_id, [])) + 1}",
            "user_id": user_id,
            "type": notification_type.value,
            "title": title,
            "message": message,
            "action_url": action_url,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "read": False,
            "read_at": None
        }
        
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        self.notifications[user_id].append(notification)
        
        logger.info(f"Notificación creada para usuario {user_id}: {title}")
        return notification
    
    def get_notifications(
        self,
        user_id: str,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Obtener notificaciones de usuario"""
        user_notifications = self.notifications.get(user_id, [])
        
        if unread_only:
            return [n for n in user_notifications if not n.get("read", False)]
        
        return user_notifications
    
    def mark_as_read(
        self,
        user_id: str,
        notification_id: str
    ) -> bool:
        """Marcar notificación como leída"""
        user_notifications = self.notifications.get(user_id, [])
        
        for notification in user_notifications:
            if notification["id"] == notification_id:
                notification["read"] = True
                notification["read_at"] = datetime.now().isoformat()
                return True
        
        return False
    
    def mark_all_as_read(self, user_id: str) -> int:
        """Marcar todas las notificaciones como leídas"""
        user_notifications = self.notifications.get(user_id, [])
        count = 0
        
        for notification in user_notifications:
            if not notification.get("read", False):
                notification["read"] = True
                notification["read_at"] = datetime.now().isoformat()
                count += 1
        
        return count
    
    def delete_notification(
        self,
        user_id: str,
        notification_id: str
    ) -> bool:
        """Eliminar notificación"""
        user_notifications = self.notifications.get(user_id, [])
        
        for i, notification in enumerate(user_notifications):
            if notification["id"] == notification_id:
                user_notifications.pop(i)
                return True
        
        return False
    
    def get_unread_count(self, user_id: str) -> int:
        """Obtener conteo de no leídas"""
        return len(self.get_notifications(user_id, unread_only=True))
    
    def create_design_ready_notification(
        self,
        user_id: str,
        store_id: str,
        store_name: str
    ) -> Dict[str, Any]:
        """Crear notificación de diseño listo"""
        return self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.SUCCESS,
            title="Diseño Completado",
            message=f"Tu diseño para '{store_name}' está listo para revisar",
            action_url=f"/designs/{store_id}",
            priority="high"
        )
    
    def create_feedback_notification(
        self,
        user_id: str,
        store_id: str,
        store_name: str
    ) -> Dict[str, Any]:
        """Crear notificación de nuevo feedback"""
        return self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.INFO,
            title="Nuevo Feedback",
            message=f"Hay nuevo feedback en tu diseño '{store_name}'",
            action_url=f"/designs/{store_id}/feedback",
            priority="normal"
        )
    
    def create_reminder_notification(
        self,
        user_id: str,
        message: str
    ) -> Dict[str, Any]:
        """Crear notificación de recordatorio"""
        return self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.REMINDER,
            title="Recordatorio",
            message=message,
            priority="normal"
        )




