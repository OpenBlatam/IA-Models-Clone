"""
Notification System for Flux2 Clothing Changer
==============================================

Advanced notification and messaging system.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Notification types."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


@dataclass
class Notification:
    """Notification information."""
    notification_id: str
    notification_type: NotificationType
    title: str
    message: str
    channels: List[NotificationChannel]
    recipient: str
    timestamp: float = time.time()
    read: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NotificationSystem:
    """Advanced notification system."""
    
    def __init__(self):
        """Initialize notification system."""
        self.notifications: Dict[str, Notification] = {}
        self.channel_handlers: Dict[NotificationChannel, Callable] = {}
        self.notification_history: deque = deque(maxlen=10000)
        
        # Statistics
        self.stats = {
            "total_notifications": 0,
            "sent_notifications": 0,
            "failed_notifications": 0,
        }
    
    def register_channel_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[Notification], bool],
    ) -> None:
        """
        Register channel handler.
        
        Args:
            channel: Notification channel
            handler: Handler function
        """
        self.channel_handlers[channel] = handler
        logger.info(f"Registered handler for channel: {channel.value}")
    
    def send_notification(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        recipient: str,
        channels: List[NotificationChannel],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Notification:
        """
        Send notification.
        
        Args:
            notification_type: Notification type
            title: Notification title
            message: Notification message
            recipient: Recipient identifier
            channels: List of channels
            metadata: Optional metadata
            
        Returns:
            Created notification
        """
        notification_id = f"notif_{int(time.time() * 1000)}"
        
        notification = Notification(
            notification_id=notification_id,
            notification_type=notification_type,
            title=title,
            message=message,
            channels=channels,
            recipient=recipient,
            metadata=metadata or {},
        )
        
        self.notifications[notification_id] = notification
        self.stats["total_notifications"] += 1
        
        # Send through channels
        success_count = 0
        for channel in channels:
            if channel in self.channel_handlers:
                try:
                    if self.channel_handlers[channel](notification):
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel.value}: {e}")
        
        if success_count > 0:
            self.stats["sent_notifications"] += 1
        else:
            self.stats["failed_notifications"] += 1
        
        self.notification_history.append(notification)
        logger.info(f"Sent notification: {notification_id}")
        
        return notification
    
    def get_notifications(
        self,
        recipient: Optional[str] = None,
        notification_type: Optional[NotificationType] = None,
        unread_only: bool = False,
    ) -> List[Notification]:
        """
        Get notifications.
        
        Args:
            recipient: Optional recipient filter
            notification_type: Optional type filter
            unread_only: Only unread notifications
            
        Returns:
            List of notifications
        """
        notifications = list(self.notifications.values())
        
        if recipient:
            notifications = [n for n in notifications if n.recipient == recipient]
        
        if notification_type:
            notifications = [n for n in notifications if n.notification_type == notification_type]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        return sorted(notifications, key=lambda n: n.timestamp, reverse=True)
    
    def mark_as_read(self, notification_id: str) -> bool:
        """
        Mark notification as read.
        
        Args:
            notification_id: Notification identifier
            
        Returns:
            True if marked
        """
        if notification_id in self.notifications:
            self.notifications[notification_id].read = True
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            **self.stats,
            "total_notifications_stored": len(self.notifications),
            "unread_notifications": len([n for n in self.notifications.values() if not n.read]),
            "channels_registered": len(self.channel_handlers),
        }


