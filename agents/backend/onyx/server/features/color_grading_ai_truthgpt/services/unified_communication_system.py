"""
Unified Communication System for Color Grading AI
==================================================

Consolidates communication services:
- WebhookManager (webhooks)
- NotificationService (notifications)
- CollaborationManager (collaboration)

Features:
- Unified interface for all communication
- Multi-channel delivery
- Event routing
- Retry logic
- Delivery tracking
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .webhook_manager import WebhookManager, Webhook
from .notification_service import NotificationService, Notification, NotificationChannel
from .collaboration_manager import CollaborationManager, ShareLink, Comment

logger = logging.getLogger(__name__)


class CommunicationChannel(Enum):
    """Communication channels."""
    WEBHOOK = "webhook"
    NOTIFICATION = "notification"
    COLLABORATION = "collaboration"
    ALL = "all"


@dataclass
class CommunicationResult:
    """Communication result."""
    channel: CommunicationChannel
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedCommunicationSystem:
    """
    Unified communication system.
    
    Consolidates:
    - WebhookManager: Webhook notifications
    - NotificationService: User notifications
    - CollaborationManager: Collaboration features
    
    Features:
    - Unified interface for all communication
    - Multi-channel delivery
    - Event routing
    - Retry logic
    - Delivery tracking
    """
    
    def __init__(self):
        """Initialize unified communication system."""
        self.webhook_manager = WebhookManager()
        self.notification_service = NotificationService()
        self.collaboration_manager = CollaborationManager()
        
        logger.info("Initialized UnifiedCommunicationSystem")
    
    async def send(
        self,
        channel: CommunicationChannel,
        message: str,
        data: Dict[str, Any],
        recipients: Optional[List[str]] = None,
        **kwargs
    ) -> CommunicationResult:
        """
        Send message through specified channel.
        
        Args:
            channel: Communication channel
            message: Message content
            data: Additional data
            recipients: Optional recipient list
            **kwargs: Additional parameters
            
        Returns:
            Communication result
        """
        if channel == CommunicationChannel.WEBHOOK or channel == CommunicationChannel.ALL:
            try:
                await self.webhook_manager.send(
                    event=kwargs.get("event", "message"),
                    data={"message": message, **data}
                )
            except Exception as e:
                logger.error(f"Webhook send error: {e}")
                if channel == CommunicationChannel.WEBHOOK:
                    return CommunicationResult(
                        channel=channel,
                        success=False,
                        error=str(e)
                    )
        
        if channel == CommunicationChannel.NOTIFICATION or channel == CommunicationChannel.ALL:
            try:
                notification = Notification(
                    title=kwargs.get("title", "Notification"),
                    message=message,
                    channel=kwargs.get("notification_channel", NotificationChannel.EMAIL),
                    recipients=recipients or [],
                    data=data
                )
                await self.notification_service.send(notification)
            except Exception as e:
                logger.error(f"Notification send error: {e}")
                if channel == CommunicationChannel.NOTIFICATION:
                    return CommunicationResult(
                        channel=channel,
                        success=False,
                        error=str(e)
                    )
        
        return CommunicationResult(
            channel=channel,
            success=True
        )
    
    def register_webhook(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None
    ):
        """
        Register webhook.
        
        Args:
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional secret for signing
        """
        webhook = Webhook(
            url=url,
            events=events,
            secret=secret
        )
        self.webhook_manager.register(webhook)
        logger.info(f"Registered webhook: {url}")
    
    async def send_notification(
        self,
        title: str,
        message: str,
        recipients: List[str],
        channel: NotificationChannel = NotificationChannel.EMAIL
    ) -> CommunicationResult:
        """
        Send notification.
        
        Args:
            title: Notification title
            message: Notification message
            recipients: Recipient list
            channel: Notification channel
            
        Returns:
            Communication result
        """
        try:
            notification = Notification(
                title=title,
                message=message,
                channel=channel,
                recipients=recipients
            )
            await self.notification_service.send(notification)
            return CommunicationResult(
                channel=CommunicationChannel.NOTIFICATION,
                success=True
            )
        except Exception as e:
            logger.error(f"Notification error: {e}")
            return CommunicationResult(
                channel=CommunicationChannel.NOTIFICATION,
                success=False,
                error=str(e)
            )
    
    def create_share_link(
        self,
        resource_id: str,
        resource_type: str,
        permissions: List[str],
        expires_at: Optional[datetime] = None
    ) -> Optional[ShareLink]:
        """
        Create share link.
        
        Args:
            resource_id: Resource ID
            resource_type: Resource type
            permissions: Permissions list
            expires_at: Optional expiration
            
        Returns:
            Share link or None
        """
        return self.collaboration_manager.create_share_link(
            resource_id=resource_id,
            resource_type=resource_type,
            permissions=permissions,
            expires_at=expires_at
        )
    
    def add_comment(
        self,
        resource_id: str,
        user_id: str,
        comment: str
    ) -> Optional[Comment]:
        """
        Add comment.
        
        Args:
            resource_id: Resource ID
            user_id: User ID
            comment: Comment text
            
        Returns:
            Comment or None
        """
        return self.collaboration_manager.add_comment(
            resource_id=resource_id,
            user_id=user_id,
            comment=comment
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            "webhooks": len(self.webhook_manager._webhooks),
            "notifications_sent": self.notification_service.get_statistics().get("sent", 0),
            "share_links": len(self.collaboration_manager._share_links),
            "comments": len(self.collaboration_manager._comments),
        }


