"""
Sistema de notificaciones mejorado
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid


class NotificationType(str, Enum):
    """Tipo de notificación"""
    ANALYSIS_READY = "analysis_ready"
    GOAL_ACHIEVED = "goal_achieved"
    REMINDER = "reminder"
    ALERT = "alert"
    COMMUNITY = "community"
    PRODUCT = "product"
    WEATHER = "weather"
    SYSTEM = "system"


@dataclass
class EnhancedNotification:
    """Notificación mejorada"""
    id: str
    user_id: str
    type: NotificationType
    title: str
    message: str
    priority: int  # 1-5, 1 = más importante
    action_url: Optional[str] = None
    action_label: Optional[str] = None
    image_url: Optional[str] = None
    read: bool = False
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type.value,
            "title": self.title,
            "message": self.message,
            "priority": self.priority,
            "action_url": self.action_url,
            "action_label": self.action_label,
            "image_url": self.image_url,
            "read": self.read,
            "created_at": self.created_at
        }


class EnhancedNotificationSystem:
    """Sistema de notificaciones mejorado"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.notifications: Dict[str, List[EnhancedNotification]] = {}  # user_id -> [notifications]
    
    def send_notification(self, user_id: str, type: NotificationType,
                         title: str, message: str, priority: int = 3,
                         action_url: Optional[str] = None,
                         action_label: Optional[str] = None,
                         image_url: Optional[str] = None) -> EnhancedNotification:
        """Envía una notificación"""
        notification = EnhancedNotification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=type,
            title=title,
            message=message,
            priority=priority,
            action_url=action_url,
            action_label=action_label,
            image_url=image_url
        )
        
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        self.notifications[user_id].append(notification)
        return notification
    
    def send_analysis_ready(self, user_id: str, analysis_id: str) -> EnhancedNotification:
        """Notifica que análisis está listo"""
        return self.send_notification(
            user_id=user_id,
            type=NotificationType.ANALYSIS_READY,
            title="Análisis Completado",
            message="Tu análisis de piel está listo. Revisa los resultados.",
            priority=2,
            action_url=f"/analysis/{analysis_id}",
            action_label="Ver Análisis"
        )
    
    def send_goal_achieved(self, user_id: str, goal_title: str) -> EnhancedNotification:
        """Notifica logro de objetivo"""
        return self.send_notification(
            user_id=user_id,
            type=NotificationType.GOAL_ACHIEVED,
            title="¡Objetivo Alcanzado!",
            message=f"Felicidades, alcanzaste tu objetivo: {goal_title}",
            priority=1,
            action_url="/goals",
            action_label="Ver Objetivos"
        )
    
    def send_weather_alert(self, user_id: str, location: str,
                          uv_index: float) -> EnhancedNotification:
        """Notifica alerta de clima"""
        return self.send_notification(
            user_id=user_id,
            type=NotificationType.WEATHER,
            title="Alerta de Clima",
            message=f"Índice UV alto ({uv_index}) en {location}. Usa protección solar.",
            priority=1,
            action_url="/weather",
            action_label="Ver Clima"
        )
    
    def get_user_notifications(self, user_id: str, unread_only: bool = False,
                              limit: int = 50) -> List[EnhancedNotification]:
        """Obtiene notificaciones del usuario"""
        user_notifications = self.notifications.get(user_id, [])
        
        if unread_only:
            user_notifications = [n for n in user_notifications if not n.read]
        
        # Ordenar por prioridad y fecha
        user_notifications.sort(
            key=lambda x: (x.priority, x.created_at),
            reverse=True
        )
        
        return user_notifications[:limit]
    
    def mark_as_read(self, user_id: str, notification_id: str) -> bool:
        """Marca notificación como leída"""
        user_notifications = self.notifications.get(user_id, [])
        
        for notification in user_notifications:
            if notification.id == notification_id:
                notification.read = True
                return True
        
        return False
    
    def mark_all_as_read(self, user_id: str) -> int:
        """Marca todas las notificaciones como leídas"""
        user_notifications = self.notifications.get(user_id, [])
        count = 0
        
        for notification in user_notifications:
            if not notification.read:
                notification.read = True
                count += 1
        
        return count
    
    def get_notification_stats(self, user_id: str) -> Dict:
        """Obtiene estadísticas de notificaciones"""
        user_notifications = self.notifications.get(user_id, [])
        
        total = len(user_notifications)
        unread = len([n for n in user_notifications if not n.read])
        
        by_type = {}
        for notification in user_notifications:
            ntype = notification.type.value
            by_type[ntype] = by_type.get(ntype, 0) + 1
        
        return {
            "total": total,
            "unread": unread,
            "read": total - unread,
            "by_type": by_type
        }






