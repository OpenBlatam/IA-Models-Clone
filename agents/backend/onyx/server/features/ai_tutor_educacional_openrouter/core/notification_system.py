"""
Notification system for student engagement and reminders.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications."""
    REMINDER = "reminder"
    ACHIEVEMENT = "achievement"
    PROGRESS = "progress"
    RECOMMENDATION = "recommendation"
    QUIZ_DUE = "quiz_due"
    STREAK_WARNING = "streak_warning"


@dataclass
class Notification:
    """Represents a notification."""
    notification_id: str
    student_id: str
    type: NotificationType
    title: str
    message: str
    priority: int = 1
    created_at: datetime = None
    read: bool = False
    read_at: Optional[datetime] = None
    action_url: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class NotificationSystem:
    """
    Manages notifications for students.
    """
    
    def __init__(self):
        self.notifications: Dict[str, List[Notification]] = {}
        self.preferences: Dict[str, Dict[str, bool]] = {}
    
    def create_notification(
        self,
        student_id: str,
        notification_type: NotificationType,
        title: str,
        message: str,
        priority: int = 1,
        action_url: Optional[str] = None
    ) -> Notification:
        """
        Create a new notification.
        
        Args:
            student_id: Student identifier
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            priority: Priority level (1-5)
            action_url: Optional URL for action
        
        Returns:
            Created notification
        """
        notification_id = f"{student_id}_{datetime.now().timestamp()}"
        
        notification = Notification(
            notification_id=notification_id,
            student_id=student_id,
            type=notification_type,
            title=title,
            message=message,
            priority=priority,
            action_url=action_url
        )
        
        if student_id not in self.notifications:
            self.notifications[student_id] = []
        
        self.notifications[student_id].append(notification)
        
        logger.info(f"Created notification {notification_id} for student {student_id}")
        
        return notification
    
    def get_notifications(
        self,
        student_id: str,
        unread_only: bool = False,
        limit: Optional[int] = None
    ) -> List[Notification]:
        """
        Get notifications for a student.
        
        Args:
            student_id: Student identifier
            unread_only: Return only unread notifications
            limit: Maximum number of notifications to return
        
        Returns:
            List of notifications
        """
        if student_id not in self.notifications:
            return []
        
        notifications = self.notifications[student_id]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        # Sort by priority and date
        notifications.sort(key=lambda x: (x.priority, x.created_at), reverse=True)
        
        if limit:
            notifications = notifications[:limit]
        
        return notifications
    
    def mark_as_read(self, student_id: str, notification_id: str):
        """Mark a notification as read."""
        if student_id not in self.notifications:
            return
        
        for notification in self.notifications[student_id]:
            if notification.notification_id == notification_id:
                notification.read = True
                notification.read_at = datetime.now()
                break
    
    def mark_all_as_read(self, student_id: str):
        """Mark all notifications as read for a student."""
        if student_id not in self.notifications:
            return
        
        for notification in self.notifications[student_id]:
            notification.read = True
            notification.read_at = datetime.now()
    
    def delete_notification(self, student_id: str, notification_id: str):
        """Delete a notification."""
        if student_id not in self.notifications:
            return
        
        self.notifications[student_id] = [
            n for n in self.notifications[student_id]
            if n.notification_id != notification_id
        ]
    
    def get_unread_count(self, student_id: str) -> int:
        """Get count of unread notifications."""
        notifications = self.get_notifications(student_id, unread_only=True)
        return len(notifications)
    
    def create_streak_reminder(self, student_id: str, days_missed: int):
        """Create a streak reminder notification."""
        if days_missed == 1:
            message = "¡No pierdas tu racha! Estudia hoy para mantener tu progreso."
        else:
            message = f"Has perdido {days_missed} días. ¡Vuelve a estudiar para recuperar tu racha!"
        
        return self.create_notification(
            student_id=student_id,
            notification_type=NotificationType.STREAK_WARNING,
            title="Recordatorio de Racha",
            message=message,
            priority=3
        )
    
    def create_achievement_notification(
        self,
        student_id: str,
        achievement_name: str,
        badge_icon: str = "🏆"
    ):
        """Create an achievement notification."""
        return self.create_notification(
            student_id=student_id,
            notification_type=NotificationType.ACHIEVEMENT,
            title=f"{badge_icon} ¡Logro Desbloqueado!",
            message=f"Has desbloqueado el logro: {achievement_name}",
            priority=5
        )
    
    def create_progress_notification(
        self,
        student_id: str,
        subject: str,
        improvement: float
    ):
        """Create a progress notification."""
        return self.create_notification(
            student_id=student_id,
            notification_type=NotificationType.PROGRESS,
            title="¡Progreso Actualizado!",
            message=f"Tu rendimiento en {subject} ha mejorado un {improvement:.1f}%",
            priority=2
        )
    
    def create_recommendation_notification(
        self,
        student_id: str,
        topic: str,
        subject: str
    ):
        """Create a recommendation notification."""
        return self.create_notification(
            student_id=student_id,
            notification_type=NotificationType.RECOMMENDATION,
            title="Nueva Recomendación",
            message=f"Te recomendamos estudiar {topic} en {subject}",
            priority=2,
            action_url=f"/study/{subject}/{topic}"
        )






