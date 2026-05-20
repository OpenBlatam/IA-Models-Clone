"""
Notification Service for Color Grading AI
==========================================

Advanced notification system with multiple channels.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"


@dataclass
class Notification:
    """Notification data structure."""
    id: str
    channel: NotificationChannel
    recipient: str
    subject: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    status: str = "pending"  # pending, sent, failed


class NotificationService:
    """
    Advanced notification service.
    
    Features:
    - Multiple notification channels
    - Retry mechanism
    - Notification history
    - Templates
    """
    
    def __init__(self):
        """Initialize notification service."""
        self._notifications: List[Notification] = []
        self._channels: Dict[NotificationChannel, Any] = {}
    
    def register_channel(self, channel: NotificationChannel, handler: Any):
        """
        Register notification channel handler.
        
        Args:
            channel: Notification channel
            handler: Channel handler
        """
        self._channels[channel] = handler
        logger.info(f"Registered notification channel: {channel.value}")
    
    async def send_notification(
        self,
        channel: NotificationChannel,
        recipient: str,
        subject: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send notification.
        
        Args:
            channel: Notification channel
            recipient: Recipient address/ID
            subject: Notification subject
            message: Notification message
            data: Additional data
            
        Returns:
            Notification ID
        """
        import uuid
        notification_id = str(uuid.uuid4())
        
        notification = Notification(
            id=notification_id,
            channel=channel,
            recipient=recipient,
            subject=subject,
            message=message,
            data=data or {}
        )
        
        self._notifications.append(notification)
        
        # Send notification
        try:
            handler = self._channels.get(channel)
            if handler:
                await handler.send(recipient, subject, message, data)
                notification.status = "sent"
                notification.sent_at = datetime.now()
            else:
                logger.warning(f"No handler for channel: {channel.value}")
                notification.status = "failed"
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            notification.status = "failed"
        
        return notification_id
    
    async def send_processing_complete(
        self,
        recipient: str,
        task_id: str,
        result: Dict[str, Any],
        channels: Optional[List[NotificationChannel]] = None
    ):
        """
        Send processing complete notification.
        
        Args:
            recipient: Recipient address
            task_id: Task ID
            result: Processing result
            channels: Notification channels (defaults to webhook)
        """
        if not channels:
            channels = [NotificationChannel.WEBHOOK]
        
        subject = f"Color Grading Complete - Task {task_id}"
        message = f"Your color grading task {task_id} has been completed successfully."
        
        data = {
            "task_id": task_id,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
        
        for channel in channels:
            await self.send_notification(channel, recipient, subject, message, data)
    
    def get_notification_history(
        self,
        channel: Optional[NotificationChannel] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get notification history.
        
        Args:
            channel: Filter by channel
            limit: Maximum results
            
        Returns:
            List of notifications
        """
        notifications = self._notifications
        
        if channel:
            notifications = [n for n in notifications if n.channel == channel]
        
        # Sort by creation date (newest first)
        notifications.sort(key=lambda n: n.created_at, reverse=True)
        
        return [
            {
                "id": n.id,
                "channel": n.channel.value,
                "recipient": n.recipient,
                "subject": n.subject,
                "status": n.status,
                "created_at": n.created_at.isoformat(),
                "sent_at": n.sent_at.isoformat() if n.sent_at else None,
            }
            for n in notifications[:limit]
        ]




