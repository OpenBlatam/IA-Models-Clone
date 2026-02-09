"""
Notification system for maintenance alerts and updates.
"""

from typing import Dict, List, Optional, Callable, Any
from ..utils.file_helpers import get_iso_timestamp
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications."""
    MAINTENANCE_DUE = "maintenance_due"
    PREDICTION_UPDATE = "prediction_update"
    ANOMALY_DETECTED = "anomaly_detected"
    SYSTEM_ALERT = "system_alert"
    CONVERSATION_UPDATE = "conversation_update"


class Notification:
    """Represents a notification."""
    
    def __init__(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        priority: str = "normal"
    ):
        self.type = notification_type
        self.title = title
        self.message = message
        self.data = data or {}
        self.priority = priority
        self.timestamp = get_iso_timestamp()
        self.read = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary."""
        return {
            "type": self.type.value,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "read": self.read
        }


class NotificationManager:
    """Manages notifications for users and systems."""
    
    def __init__(self):
        self.notifications: Dict[str, List[Notification]] = {}
        self.handlers: Dict[NotificationType, List[Callable]] = {}
    
    def subscribe(
        self,
        notification_type: NotificationType,
        handler: Callable[[Notification], None]
    ):
        """
        Subscribe to a notification type.
        
        Args:
            notification_type: Type of notification
            handler: Function to call when notification is sent
        """
        if notification_type not in self.handlers:
            self.handlers[notification_type] = []
        self.handlers[notification_type].append(handler)
        logger.info(f"Subscribed to {notification_type.value}")
    
    def send_notification(
        self,
        user_id: str,
        notification: Notification
    ):
        """
        Send a notification to a user.
        
        Args:
            user_id: User identifier
            notification: Notification object
        """
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        self.notifications[user_id].append(notification)
        
        if notification.type in self.handlers:
            for handler in self.handlers[notification.type]:
                try:
                    handler(notification)
                except Exception as e:
                    logger.error(f"Error in notification handler: {e}")
        
        logger.info(f"Notification sent to {user_id}: {notification.title}")
    
    def get_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Notification]:
        """
        Get notifications for a user.
        
        Args:
            user_id: User identifier
            unread_only: Only return unread notifications
            limit: Maximum number of notifications to return
        
        Returns:
            List of notifications
        """
        if user_id not in self.notifications:
            return []
        
        notifications = self.notifications[user_id]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        return sorted(notifications, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def mark_as_read(self, user_id: str, notification_index: int):
        """
        Mark a notification as read.
        
        Args:
            user_id: User identifier
            notification_index: Index of notification
        """
        if user_id in self.notifications:
            notifications = self.notifications[user_id]
            if 0 <= notification_index < len(notifications):
                notifications[notification_index].read = True
    
    def clear_notifications(self, user_id: str):
        """
        Clear all notifications for a user.
        
        Args:
            user_id: User identifier
        """
        if user_id in self.notifications:
            del self.notifications[user_id]
            logger.info(f"Notifications cleared for {user_id}")






