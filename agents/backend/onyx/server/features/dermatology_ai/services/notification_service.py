"""
Sistema de notificaciones (Email, Push, SMS)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json


class NotificationType(str, Enum):
    """Tipos de notificación"""
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"
    IN_APP = "in_app"


class NotificationPriority(str, Enum):
    """Prioridad de notificación"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notificación"""
    id: str
    user_id: str
    type: NotificationType
    title: str
    message: str
    priority: NotificationPriority
    data: Optional[Dict] = None
    sent: bool = False
    sent_at: Optional[str] = None
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
            "priority": self.priority.value,
            "data": self.data or {},
            "sent": self.sent,
            "sent_at": self.sent_at,
            "created_at": self.created_at
        }


class NotificationService:
    """Servicio de notificaciones"""
    
    def __init__(self):
        """Inicializa el servicio de notificaciones"""
        self.notifications: Dict[str, List[Notification]] = {}  # user_id -> [notifications]
        self.email_enabled = False  # Configurar según necesidad
        self.push_enabled = False
        self.sms_enabled = False
    
    def send_notification(self, user_id: str, notification_type: NotificationType,
                         title: str, message: str,
                         priority: NotificationPriority = NotificationPriority.NORMAL,
                         data: Optional[Dict] = None) -> str:
        """
        Envía una notificación
        
        Args:
            user_id: ID del usuario
            notification_type: Tipo de notificación
            title: Título
            message: Mensaje
            priority: Prioridad
            data: Datos adicionales
            
        Returns:
            ID de la notificación
        """
        import hashlib
        notification_id = hashlib.md5(
            f"{user_id}{title}{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        notification = Notification(
            id=notification_id,
            user_id=user_id,
            type=notification_type,
            title=title,
            message=message,
            priority=priority,
            data=data
        )
        
        # Enviar según tipo
        if notification_type == NotificationType.EMAIL and self.email_enabled:
            self._send_email(notification)
        elif notification_type == NotificationType.PUSH and self.push_enabled:
            self._send_push(notification)
        elif notification_type == NotificationType.SMS and self.sms_enabled:
            self._send_sms(notification)
        else:
            # Guardar como in-app
            notification.type = NotificationType.IN_APP
        
        # Guardar notificación
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        self.notifications[user_id].append(notification)
        
        return notification_id
    
    def send_analysis_complete_notification(self, user_id: str, analysis_result: Dict):
        """Envía notificación de análisis completado"""
        overall_score = analysis_result.get("quality_scores", {}).get("overall_score", 0)
        
        if overall_score >= 80:
            title = "¡Excelente! Tu análisis está listo"
            message = f"Tu score de piel es {overall_score:.1f}/100. ¡Sigue así!"
            priority = NotificationPriority.NORMAL
        elif overall_score >= 60:
            title = "Análisis completado"
            message = f"Tu score es {overall_score:.1f}/100. Revisa las recomendaciones."
            priority = NotificationPriority.NORMAL
        else:
            title = "Análisis completado - Atención requerida"
            message = f"Tu score es {overall_score:.1f}/100. Revisa las alertas y recomendaciones."
            priority = NotificationPriority.HIGH
        
        return self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.IN_APP,
            title=title,
            message=message,
            priority=priority,
            data={"analysis_result": analysis_result}
        )
    
    def send_alert_notification(self, user_id: str, alert: Dict):
        """Envía notificación de alerta"""
        level = alert.get("level", "info")
        priority_map = {
            "critical": NotificationPriority.URGENT,
            "warning": NotificationPriority.HIGH,
            "info": NotificationPriority.NORMAL
        }
        
        return self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.IN_APP,
            title=alert.get("title", "Nueva alerta"),
            message=alert.get("message", ""),
            priority=priority_map.get(level, NotificationPriority.NORMAL),
            data={"alert": alert}
        )
    
    def send_progress_milestone_notification(self, user_id: str, milestone: str):
        """Envía notificación de hito de progreso"""
        return self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.IN_APP,
            title="¡Hito alcanzado!",
            message=milestone,
            priority=NotificationPriority.NORMAL,
            data={"milestone": milestone}
        )
    
    def get_user_notifications(self, user_id: str, unread_only: bool = False,
                              limit: int = 50) -> List[Notification]:
        """
        Obtiene notificaciones de un usuario
        
        Args:
            user_id: ID del usuario
            unread_only: Solo no leídas
            limit: Límite de resultados
            
        Returns:
            Lista de notificaciones
        """
        notifications = self.notifications.get(user_id, [])
        
        if unread_only:
            notifications = [n for n in notifications if not n.sent]
        
        # Ordenar por fecha (más reciente primero)
        notifications.sort(key=lambda x: x.created_at, reverse=True)
        
        return notifications[:limit]
    
    def mark_as_read(self, user_id: str, notification_id: str):
        """Marca una notificación como leída"""
        notifications = self.notifications.get(user_id, [])
        for notification in notifications:
            if notification.id == notification_id:
                notification.sent = True
                notification.sent_at = datetime.now().isoformat()
                break
    
    def get_unread_count(self, user_id: str) -> int:
        """Obtiene cantidad de notificaciones no leídas"""
        notifications = self.notifications.get(user_id, [])
        return len([n for n in notifications if not n.sent])
    
    def _send_email(self, notification: Notification):
        """Envía email (placeholder)"""
        # Implementar con librería de email (smtplib, sendgrid, etc.)
        notification.sent = True
        notification.sent_at = datetime.now().isoformat()
    
    def _send_push(self, notification: Notification):
        """Envía push notification (placeholder)"""
        # Implementar con FCM, OneSignal, etc.
        notification.sent = True
        notification.sent_at = datetime.now().isoformat()
    
    def _send_sms(self, notification: Notification):
        """Envía SMS (placeholder)"""
        # Implementar con Twilio, AWS SNS, etc.
        notification.sent = True
        notification.sent_at = datetime.now().isoformat()






