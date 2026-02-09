"""
Real-time Notifications System
===============================

Real-time notification system with multiple channels.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification channel."""
    WEBSOCKET = "websocket"
    SSE = "sse"
    EMAIL = "email"
    PUSH = "push"
    IN_APP = "in_app"


class NotificationPriority(Enum):
    """Notification priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notification."""
    id: str
    user_id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority = NotificationPriority.NORMAL
    created_at: float = 0.0
    read_at: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def mark_as_read(self) -> None:
        """Mark notification as read."""
        self.read_at = time.time()


class RealTimeNotifications:
    """Real-time notification system."""
    
    def __init__(self):
        """Initialize notification system."""
        self.notifications: Dict[str, List[Notification]] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.channel_handlers: Dict[NotificationChannel, Callable] = {}
    
    def subscribe(
        self,
        user_id: str,
        callback: Callable,
    ) -> None:
        """
        Subscribe user to notifications.
        
        Args:
            user_id: User ID
            callback: Callback function
        """
        if user_id not in self.subscribers:
            self.subscribers[user_id] = []
        
        self.subscribers[user_id].append(callback)
        logger.debug(f"User subscribed: {user_id}")
    
    def unsubscribe(
        self,
        user_id: str,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Unsubscribe user from notifications.
        
        Args:
            user_id: User ID
            callback: Optional specific callback
        """
        if user_id not in self.subscribers:
            return
        
        if callback:
            self.subscribers[user_id].remove(callback)
        else:
            del self.subscribers[user_id]
        
        logger.debug(f"User unsubscribed: {user_id}")
    
    def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        channel: NotificationChannel = NotificationChannel.IN_APP,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Notification:
        """
        Send notification to user.
        
        Args:
            user_id: User ID
            title: Notification title
            message: Notification message
            channel: Notification channel
            priority: Notification priority
            metadata: Optional metadata
            
        Returns:
            Created notification
        """
        notification_id = f"notif_{int(time.time() * 1000)}"
        
        notification = Notification(
            id=notification_id,
            user_id=user_id,
            title=title,
            message=message,
            channel=channel,
            priority=priority,
            metadata=metadata or {},
        )
        
        # Store notification
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        self.notifications[user_id].append(notification)
        
        # Keep only last 100 notifications per user
        if len(self.notifications[user_id]) > 100:
            self.notifications[user_id] = self.notifications[user_id][-100:]
        
        # Notify subscribers
        if user_id in self.subscribers:
            for callback in self.subscribers[user_id]:
                try:
                    callback(notification)
                except Exception as e:
                    logger.error(f"Notification callback error: {e}")
        
        # Use channel handler if available
        if channel in self.channel_handlers:
            try:
                self.channel_handlers[channel](notification)
            except Exception as e:
                logger.error(f"Channel handler error: {e}")
        
        logger.info(f"Notification sent: {user_id} - {title}")
        
        return notification
    
    def get_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50,
    ) -> List[Notification]:
        """
        Get notifications for user.
        
        Args:
            user_id: User ID
            unread_only: Only unread notifications
            limit: Maximum number of results
            
        Returns:
            List of notifications
        """
        if user_id not in self.notifications:
            return []
        
        notifications = self.notifications[user_id]
        
        if unread_only:
            notifications = [n for n in notifications if n.read_at is None]
        
        return notifications[-limit:]
    
    def mark_as_read(
        self,
        user_id: str,
        notification_id: str,
    ) -> bool:
        """
        Mark notification as read.
        
        Args:
            user_id: User ID
            notification_id: Notification ID
            
        Returns:
            True if marked
        """
        if user_id not in self.notifications:
            return False
        
        for notification in self.notifications[user_id]:
            if notification.id == notification_id:
                notification.mark_as_read()
                return True
        
        return False
    
    def mark_all_as_read(self, user_id: str) -> int:
        """
        Mark all notifications as read for user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of notifications marked
        """
        if user_id not in self.notifications:
            return 0
        
        count = 0
        for notification in self.notifications[user_id]:
            if notification.read_at is None:
                notification.mark_as_read()
                count += 1
        
        return count
    
    def register_channel_handler(
        self,
        channel: NotificationChannel,
        handler: Callable,
    ) -> None:
        """
        Register channel handler.
        
        Args:
            channel: Notification channel
            handler: Handler function
        """
        self.channel_handlers[channel] = handler
        logger.info(f"Channel handler registered: {channel.value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification system statistics."""
        total_notifications = sum(len(n) for n in self.notifications.values())
        unread_count = sum(
            len([n for n in notifications if n.read_at is None])
            for notifications in self.notifications.values()
        )
        
        return {
            "total_users": len(self.notifications),
            "total_notifications": total_notifications,
            "unread_notifications": unread_count,
            "subscribers": len(self.subscribers),
            "channel_handlers": len(self.channel_handlers),
        }

