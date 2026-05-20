"""
Real-time Notifications System
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Notification:
    """Notification object"""
    
    def __init__(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize notification
        
        Args:
            title: Notification title
            message: Notification message
            level: Notification level
            metadata: Optional metadata
        """
        self.title = title
        self.message = message
        self.level = level
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.read = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "message": self.message,
            "level": self.level.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "read": self.read
        }


class NotificationManager:
    """Manage notifications"""
    
    def __init__(self, max_notifications: int = 1000):
        """
        Initialize notification manager
        
        Args:
            max_notifications: Maximum notifications to store
        """
        self.notifications: List[Notification] = []
        self.max_notifications = max_notifications
        self.subscribers: List[Callable] = []
        
        logger.info("NotificationManager initialized")
    
    def send_notification(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Send notification
        
        Args:
            title: Notification title
            message: Notification message
            level: Notification level
            metadata: Optional metadata
        """
        notification = Notification(title, message, level, metadata)
        
        # Add to list
        self.notifications.append(notification)
        
        # Limit size
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(notification)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")
        
        logger.info(f"Notification sent: {title} ({level.value})")
    
    def subscribe(self, callback: Callable):
        """Subscribe to notifications"""
        self.subscribers.append(callback)
    
    def get_notifications(
        self,
        level: Optional[NotificationLevel] = None,
        unread_only: bool = False,
        limit: Optional[int] = None
    ) -> List[Notification]:
        """
        Get notifications
        
        Args:
            level: Optional level filter
            unread_only: Only unread notifications
            limit: Optional limit
        
        Returns:
            List of notifications
        """
        notifications = self.notifications
        
        if level:
            notifications = [n for n in notifications if n.level == level]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        if limit:
            notifications = notifications[-limit:]
        
        return notifications
    
    def mark_read(self, notification_index: int):
        """Mark notification as read"""
        if 0 <= notification_index < len(self.notifications):
            self.notifications[notification_index].read = True
    
    def clear_notifications(self):
        """Clear all notifications"""
        self.notifications.clear()


class AlertSystem:
    """Alert system for critical events"""
    
    def __init__(self, notification_manager: NotificationManager):
        """
        Initialize alert system
        
        Args:
            notification_manager: Notification manager
        """
        self.notification_manager = notification_manager
        self.alert_rules = []
    
    def add_alert_rule(
        self,
        condition: Callable,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.WARNING
    ):
        """
        Add alert rule
        
        Args:
            condition: Condition function that returns True to trigger
            title: Alert title
            message: Alert message
            level: Alert level
        """
        self.alert_rules.append({
            "condition": condition,
            "title": title,
            "message": message,
            "level": level
        })
    
    def check_alerts(self, context: Dict[str, Any]):
        """Check all alert rules"""
        for rule in self.alert_rules:
            try:
                if rule["condition"](context):
                    self.notification_manager.send_notification(
                        title=rule["title"],
                        message=rule["message"],
                        level=rule["level"]
                    )
            except Exception as e:
                logger.error(f"Alert check failed: {e}")

