"""
Notification Service
====================
Service for sending notifications via multiple channels
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"


class NotificationPriority(Enum):
    """Notification priority"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notification"""
    id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority = NotificationPriority.NORMAL
    recipient: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    sent_at: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None


class NotificationService:
    """
    Service for sending notifications via multiple channels
    """
    
    def __init__(self):
        self._notifications: Dict[str, Notification] = {}
        self._handlers: Dict[NotificationChannel, Callable] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._stats = {
            'sent': 0,
            'failed': 0,
            'pending': 0
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def register_handler(
        self,
        channel: NotificationChannel,
        handler: Callable
    ):
        """Register handler for notification channel"""
        self._handlers[channel] = handler
    
    async def send(
        self,
        title: str,
        message: str,
        channel: NotificationChannel,
        recipient: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Send notification"""
        notification_id = f"notif_{int(time.time() * 1000)}"
        
        notification = Notification(
            id=notification_id,
            title=title,
            message=message,
            channel=channel,
            priority=priority,
            recipient=recipient,
            metadata=metadata or {}
        )
        
        self._notifications[notification_id] = notification
        self._stats['pending'] += 1
        
        # Send notification
        try:
            handler = self._handlers.get(channel)
            if handler:
                await handler(notification)
            else:
                # Default handler
                await self._default_handler(notification)
            
            notification.status = "sent"
            notification.sent_at = time.time()
            self._stats['sent'] += 1
            self._stats['pending'] -= 1
            
        except Exception as e:
            notification.status = "failed"
            notification.error = str(e)
            self._stats['failed'] += 1
            self._stats['pending'] -= 1
            logger.error(f"Failed to send notification {notification_id}: {e}")
        
        return notification
    
    async def _default_handler(self, notification: Notification):
        """Default notification handler"""
        logger.info(
            f"Notification [{notification.channel.value}] "
            f"to {notification.recipient}: {notification.title}"
        )
        # In production, implement actual sending logic
    
    async def send_batch(
        self,
        notifications: List[Dict[str, Any]]
    ) -> List[Notification]:
        """Send multiple notifications"""
        results = []
        for notif_data in notifications:
            notification = await self.send(**notif_data)
            results.append(notification)
        return results
    
    async def send_to_multiple_channels(
        self,
        title: str,
        message: str,
        channels: List[NotificationChannel],
        recipient: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> List[Notification]:
        """Send notification to multiple channels"""
        notifications = []
        for channel in channels:
            notification = await self.send(
                title=title,
                message=message,
                channel=channel,
                recipient=recipient,
                priority=priority
            )
            notifications.append(notification)
        return notifications
    
    def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Get notification by ID"""
        return self._notifications.get(notification_id)
    
    def list_notifications(
        self,
        channel: Optional[NotificationChannel] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Notification]:
        """List notifications with filters"""
        notifications = list(self._notifications.values())
        
        if channel:
            notifications = [n for n in notifications if n.channel == channel]
        
        if status:
            notifications = [n for n in notifications if n.status == status]
        
        # Sort by created_at descending
        notifications.sort(key=lambda n: n.created_at, reverse=True)
        
        return notifications[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        channel_counts = {}
        priority_counts = {}
        status_counts = {}
        
        for notification in self._notifications.values():
            channel = notification.channel.value
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
            
            priority = notification.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            status = notification.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_notifications': len(self._notifications),
            'sent': self._stats['sent'],
            'failed': self._stats['failed'],
            'pending': self._stats['pending'],
            'by_channel': channel_counts,
            'by_priority': priority_counts,
            'by_status': status_counts
        }


# Global notification service instance
notification_service = NotificationService()

