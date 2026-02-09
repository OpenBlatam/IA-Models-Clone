"""
Notifications Service - Sistema de notificaciones
==================================================

Sistema completo de notificaciones y alertas para mantener al usuario
informado y motivado.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Tipos de notificaciones"""
    ACHIEVEMENT = "achievement"
    BADGE_UNLOCKED = "badge_unlocked"
    LEVEL_UP = "level_up"
    NEW_JOB_MATCH = "new_job_match"
    JOB_APPLICATION_UPDATE = "job_application_update"
    STREAK_REMINDER = "streak_reminder"
    STEP_REMINDER = "step_reminder"
    SKILL_RECOMMENDATION = "skill_recommendation"
    COMMUNITY_MESSAGE = "community_message"
    MENTOR_MESSAGE = "mentor_message"
    DAILY_TIP = "daily_tip"
    MOTIVATIONAL = "motivational"


class NotificationPriority(str, Enum):
    """Prioridad de notificaciones"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Representa una notificación"""
    id: str
    user_id: str
    type: NotificationType
    title: str
    message: str
    priority: NotificationPriority
    created_at: datetime
    read: bool = False
    read_at: Optional[datetime] = None
    action_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationsService:
    """Servicio de notificaciones"""
    
    def __init__(self):
        """Inicializar servicio de notificaciones"""
        self.notifications: Dict[str, List[Notification]] = {}
        logger.info("NotificationsService initialized")
    
    def create_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        action_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Crear una nueva notificación"""
        notification = Notification(
            id=f"notif_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            type=notification_type,
            title=title,
            message=message,
            priority=priority,
            created_at=datetime.now(),
            action_url=action_url,
            metadata=metadata or {}
        )
        
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        self.notifications[user_id].append(notification)
        
        logger.info(f"Notification created for user {user_id}: {title}")
        return notification
    
    def notify_level_up(self, user_id: str, new_level: int) -> Notification:
        """Notificar subida de nivel"""
        return self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.LEVEL_UP,
            title=f"🎉 ¡Subiste al nivel {new_level}!",
            message=f"Felicidades, has alcanzado el nivel {new_level}. ¡Sigue así!",
            priority=NotificationPriority.HIGH,
            action_url=f"/dashboard?level={new_level}",
            metadata={"level": new_level}
        )
    
    def notify_badge_unlocked(self, user_id: str, badge_name: str, badge_icon: str) -> Notification:
        """Notificar badge desbloqueado"""
        return self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.BADGE_UNLOCKED,
            title=f"{badge_icon} Badge desbloqueado: {badge_name}",
            message=f"Has desbloqueado el badge '{badge_name}'. ¡Excelente trabajo!",
            priority=NotificationPriority.HIGH,
            action_url="/badges",
            metadata={"badge_name": badge_name, "badge_icon": badge_icon}
        )
    
    def notify_new_job_match(self, user_id: str, job_title: str, company: str) -> Notification:
        """Notificar nuevo match de trabajo"""
        return self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.NEW_JOB_MATCH,
            title="💼 ¡Nuevo match de trabajo!",
            message=f"{company} está interesado en tu perfil para {job_title}",
            priority=NotificationPriority.HIGH,
            action_url="/jobs/matches",
            metadata={"job_title": job_title, "company": company}
        )
    
    def notify_streak_reminder(self, user_id: str, current_streak: int) -> Notification:
        """Recordatorio de racha"""
        return self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.STREAK_REMINDER,
            title=f"🔥 Racha de {current_streak} días",
            message=f"Llevas {current_streak} días consecutivos. ¡No rompas la racha!",
            priority=NotificationPriority.MEDIUM,
            action_url="/dashboard",
            metadata={"streak": current_streak}
        )
    
    def notify_step_reminder(self, user_id: str, step_title: str) -> Notification:
        """Recordatorio de paso pendiente"""
        return self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.STEP_REMINDER,
            title="📋 Paso pendiente",
            message=f"Recuerda completar: {step_title}",
            priority=NotificationPriority.MEDIUM,
            action_url="/steps",
            metadata={"step_title": step_title}
        )
    
    def notify_daily_tip(self, user_id: str, tip: str) -> Notification:
        """Consejo diario"""
        return self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.DAILY_TIP,
            title="💡 Consejo del día",
            message=tip,
            priority=NotificationPriority.LOW,
            action_url="/tips"
        )
    
    def notify_motivational(self, user_id: str, message: str) -> Notification:
        """Mensaje motivacional"""
        motivational_messages = [
            "¡Tú puedes hacerlo! Cada paso te acerca a tu objetivo.",
            "El éxito es la suma de pequeños esfuerzos repetidos día tras día.",
            "No te rindas. Los mejores momentos están por venir.",
            "Cada 'no' te acerca más a un 'sí'. Sigue intentando.",
            "Tu futuro self te agradecerá por el esfuerzo de hoy.",
        ]
        
        return self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.MOTIVATIONAL,
            title="🌟 Mensaje motivacional",
            message=message or motivational_messages[datetime.now().day % len(motivational_messages)],
            priority=NotificationPriority.LOW,
            action_url="/dashboard"
        )
    
    def get_user_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: Optional[int] = None
    ) -> List[Notification]:
        """Obtener notificaciones del usuario"""
        if user_id not in self.notifications:
            return []
        
        notifications = self.notifications[user_id]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        # Ordenar por fecha (más recientes primero)
        notifications.sort(key=lambda x: x.created_at, reverse=True)
        
        if limit:
            notifications = notifications[:limit]
        
        return notifications
    
    def mark_as_read(self, user_id: str, notification_id: str) -> bool:
        """Marcar notificación como leída"""
        if user_id not in self.notifications:
            return False
        
        for notification in self.notifications[user_id]:
            if notification.id == notification_id:
                notification.read = True
                notification.read_at = datetime.now()
                return True
        
        return False
    
    def mark_all_as_read(self, user_id: str) -> int:
        """Marcar todas las notificaciones como leídas"""
        if user_id not in self.notifications:
            return 0
        
        count = 0
        for notification in self.notifications[user_id]:
            if not notification.read:
                notification.read = True
                notification.read_at = datetime.now()
                count += 1
        
        return count
    
    def get_unread_count(self, user_id: str) -> int:
        """Obtener cantidad de notificaciones no leídas"""
        notifications = self.get_user_notifications(user_id, unread_only=True)
        return len(notifications)
    
    def delete_notification(self, user_id: str, notification_id: str) -> bool:
        """Eliminar notificación"""
        if user_id not in self.notifications:
            return False
        
        self.notifications[user_id] = [
            n for n in self.notifications[user_id]
            if n.id != notification_id
        ]
        
        return True




