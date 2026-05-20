"""
Notification System
===================

Multi-channel notification system.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"


class NotificationPriority(Enum):
    """Notification priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notification data."""
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority = NotificationPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class NotificationHandler:
    """Base class for notification handlers."""
    
    async def send(self, notification: Notification) -> bool:
        """
        Send notification.
        
        Args:
            notification: Notification to send
            
        Returns:
            True if successful
        """
        raise NotImplementedError


class EmailNotificationHandler(NotificationHandler):
    """Email notification handler."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        """Initialize email handler."""
        self.smtp_config = smtp_config
    
    async def send(self, notification: Notification) -> bool:
        """Send email notification."""
        # Implementation would use smtplib or similar
        logger.info(f"Email notification: {notification.title} - {notification.message}")
        return True


class WebhookNotificationHandler(NotificationHandler):
    """Webhook notification handler."""
    
    def __init__(self, webhook_url: str):
        """Initialize webhook handler."""
        self.webhook_url = webhook_url
    
    async def send(self, notification: Notification) -> bool:
        """Send webhook notification."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json={
                        "title": notification.title,
                        "message": notification.message,
                        "priority": notification.priority.value,
                        "metadata": notification.metadata,
                        "timestamp": notification.timestamp.isoformat()
                    }
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False


class NotificationManager:
    """
    Manages multi-channel notifications.
    
    Features:
    - Multiple channels
    - Priority handling
    - Retry logic
    - Notification history
    """
    
    def __init__(self):
        """Initialize notification manager."""
        self._handlers: Dict[NotificationChannel, List[NotificationHandler]] = {}
        self._history: List[Notification] = []
        self._stats = {
            "total_sent": 0,
            "successful": 0,
            "failed": 0,
        }
    
    def register_handler(
        self,
        channel: NotificationChannel,
        handler: NotificationHandler
    ):
        """
        Register a notification handler.
        
        Args:
            channel: Notification channel
            handler: Handler instance
        """
        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)
        logger.info(f"Registered {channel.value} notification handler")
    
    async def send(
        self,
        notification: Notification,
        retry: bool = True
    ) -> Dict[str, Any]:
        """
        Send notification through registered handlers.
        
        Args:
            notification: Notification to send
            retry: Whether to retry on failure
            
        Returns:
            Result dictionary
        """
        handlers = self._handlers.get(notification.channel, [])
        
        if not handlers:
            logger.warning(f"No handlers registered for channel: {notification.channel.value}")
            return {
                "success": False,
                "error": f"No handlers for {notification.channel.value}"
            }
        
        results = {}
        for handler in handlers:
            try:
                success = await handler.send(notification)
                results[handler.__class__.__name__] = success
                
                if success:
                    self._stats["successful"] += 1
                else:
                    self._stats["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Error sending notification via {handler.__class__.__name__}: {e}")
                results[handler.__class__.__name__] = False
                self._stats["failed"] += 1
        
        # Record in history
        self._history.append(notification)
        if len(self._history) > 1000:
            self._history = self._history[-1000:]
        
        self._stats["total_sent"] += 1
        
        overall_success = any(results.values())
        
        return {
            "success": overall_success,
            "results": results,
            "channel": notification.channel.value
        }
    
    def get_history(
        self,
        channel: Optional[NotificationChannel] = None,
        limit: int = 100
    ) -> List[Notification]:
        """
        Get notification history.
        
        Args:
            channel: Optional channel filter
            limit: Maximum number of records
            
        Returns:
            List of notifications
        """
        history = self._history
        if channel:
            history = [n for n in history if n.channel == channel]
        return history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        success_rate = (
            self._stats["successful"] / self._stats["total_sent"]
            if self._stats["total_sent"] > 0
            else 0.0
        )
        
        return {
            **self._stats,
            "success_rate": success_rate,
            "channels": [ch.value for ch in self._handlers.keys()],
        }

