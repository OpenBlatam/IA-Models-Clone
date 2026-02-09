"""
Notification System - Sistema de notificaciones
===============================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Tipos de notificaciones"""
    PROTOTYPE_GENERATED = "prototype_generated"
    VALIDATION_WARNING = "validation_warning"
    COST_ALERT = "cost_alert"
    MATERIAL_UNAVAILABLE = "material_unavailable"
    FEASIBILITY_LOW = "feasibility_low"
    VERSION_CREATED = "version_created"
    SYSTEM_UPDATE = "system_update"


@dataclass
class Notification:
    """Notificación"""
    id: str
    type: NotificationType
    title: str
    message: str
    severity: str  # info, warning, error, success
    user_id: Optional[str] = None
    prototype_id: Optional[str] = None
    created_at: datetime = None
    read: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class NotificationSystem:
    """Sistema de notificaciones"""
    
    def __init__(self):
        self.notifications: Dict[str, List[Notification]] = {}
        self.preferences: Dict[str, Dict[str, bool]] = {}
    
    def send_notification(self, notification: Notification):
        """Envía una notificación"""
        user_id = notification.user_id or "system"
        
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        self.notifications[user_id].append(notification)
        
        # Limpiar notificaciones antiguas (más de 30 días)
        cutoff = datetime.now() - timedelta(days=30)
        self.notifications[user_id] = [
            n for n in self.notifications[user_id]
            if n.created_at > cutoff
        ]
        
        logger.info(f"Notificación enviada: {notification.type} a {user_id}")
    
    def notify_prototype_generated(self, user_id: str, prototype_id: str, 
                                   product_name: str):
        """Notifica generación de prototipo"""
        notification = Notification(
            id=f"proto_{prototype_id}",
            type=NotificationType.PROTOTYPE_GENERATED,
            title="Prototipo Generado",
            message=f"Tu prototipo '{product_name}' ha sido generado exitosamente",
            severity="success",
            user_id=user_id,
            prototype_id=prototype_id
        )
        self.send_notification(notification)
    
    def notify_validation_warning(self, user_id: str, prototype_id: str,
                                  issues: List[Dict[str, Any]]):
        """Notifica advertencias de validación"""
        num_issues = len(issues)
        notification = Notification(
            id=f"val_{prototype_id}",
            type=NotificationType.VALIDATION_WARNING,
            title="Advertencias de Validación",
            message=f"Se encontraron {num_issues} problemas en tu prototipo",
            severity="warning",
            user_id=user_id,
            prototype_id=prototype_id
        )
        self.send_notification(notification)
    
    def notify_cost_alert(self, user_id: str, prototype_id: str, cost: float,
                         budget: Optional[float] = None):
        """Notifica alerta de costo"""
        if budget and cost > budget * 1.2:
            notification = Notification(
                id=f"cost_{prototype_id}",
                type=NotificationType.COST_ALERT,
                title="Alerta de Costo",
                message=f"El costo (${cost:.2f}) excede tu presupuesto en más del 20%",
                severity="warning",
                user_id=user_id,
                prototype_id=prototype_id
            )
            self.send_notification(notification)
    
    def notify_feasibility_low(self, user_id: str, prototype_id: str, score: float):
        """Notifica viabilidad baja"""
        if score < 40:
            notification = Notification(
                id=f"feas_{prototype_id}",
                type=NotificationType.FEASIBILITY_LOW,
                title="Viabilidad Baja",
                message=f"El prototipo tiene una viabilidad baja ({score}/100). Considera simplificarlo",
                severity="warning",
                user_id=user_id,
                prototype_id=prototype_id
            )
            self.send_notification(notification)
    
    def get_notifications(self, user_id: str, unread_only: bool = False,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene notificaciones de un usuario"""
        user_notifications = self.notifications.get(user_id, [])
        
        if unread_only:
            user_notifications = [n for n in user_notifications if not n.read]
        
        # Ordenar por fecha (más reciente primero)
        user_notifications.sort(key=lambda x: x.created_at, reverse=True)
        
        return [
            {
                "id": n.id,
                "type": n.type.value,
                "title": n.title,
                "message": n.message,
                "severity": n.severity,
                "prototype_id": n.prototype_id,
                "created_at": n.created_at.isoformat(),
                "read": n.read
            }
            for n in user_notifications[:limit]
        ]
    
    def mark_as_read(self, user_id: str, notification_id: str) -> bool:
        """Marca una notificación como leída"""
        user_notifications = self.notifications.get(user_id, [])
        for notification in user_notifications:
            if notification.id == notification_id:
                notification.read = True
                return True
        return False
    
    def mark_all_as_read(self, user_id: str):
        """Marca todas las notificaciones como leídas"""
        user_notifications = self.notifications.get(user_id, [])
        for notification in user_notifications:
            notification.read = True
    
    def get_unread_count(self, user_id: str) -> int:
        """Obtiene el conteo de notificaciones no leídas"""
        user_notifications = self.notifications.get(user_id, [])
        return sum(1 for n in user_notifications if not n.read)
    
    def delete_notification(self, user_id: str, notification_id: str) -> bool:
        """Elimina una notificación"""
        user_notifications = self.notifications.get(user_id, [])
        self.notifications[user_id] = [
            n for n in user_notifications if n.id != notification_id
        ]
        return len(self.notifications[user_id]) < len(user_notifications)
    
    def set_preferences(self, user_id: str, preferences: Dict[str, bool]):
        """Configura preferencias de notificaciones"""
        self.preferences[user_id] = preferences
    
    def should_notify(self, user_id: str, notification_type: NotificationType) -> bool:
        """Verifica si se debe notificar según preferencias"""
        user_prefs = self.preferences.get(user_id, {})
        return user_prefs.get(notification_type.value, True)




