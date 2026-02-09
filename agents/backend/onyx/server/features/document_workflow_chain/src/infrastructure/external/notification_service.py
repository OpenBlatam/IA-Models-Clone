"""
Notification Service
====================

Advanced notification service for sending notifications via multiple channels.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from ...shared.events.event_bus import get_event_bus, DomainEvent, EventMetadata


logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    """Notification channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"


class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class NotificationConfig:
    """Notification service configuration"""
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = True
    webhook_enabled: bool = False
    slack_enabled: bool = False
    teams_enabled: bool = False
    discord_enabled: bool = False
    
    # Email settings
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    
    # SMS settings
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    
    # Push settings
    firebase_server_key: str = ""
    
    # Webhook settings
    webhook_url: str = ""
    webhook_secret: str = ""
    
    # Slack settings
    slack_webhook_url: str = ""
    slack_bot_token: str = ""
    
    # Teams settings
    teams_webhook_url: str = ""
    
    # Discord settings
    discord_webhook_url: str = ""


@dataclass
class Notification:
    """Notification data"""
    id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority
    recipient: str
    metadata: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


class NotificationProvider(ABC):
    """Abstract notification provider interface"""
    
    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send notification"""
        pass
    
    @abstractmethod
    async def validate_recipient(self, recipient: str) -> bool:
        """Validate recipient format"""
        pass


class EmailProvider(NotificationProvider):
    """Email notification provider"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize email client"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            self._smtplib = smtplib
            self._MIMEText = MIMEText
            self._MIMEMultipart = MIMEMultipart
        except ImportError:
            logger.error("Email libraries not available")
            raise
    
    async def send(self, notification: Notification) -> bool:
        """Send email notification"""
        try:
            # Create message
            msg = self._MIMEMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = notification.recipient
            msg['Subject'] = notification.title
            
            # Add body
            msg.attach(self._MIMEText(notification.message, 'html'))
            
            # Send email
            with self._smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.smtp_use_tls:
                    server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {notification.recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Email send failed to {notification.recipient}: {e}")
            return False
    
    async def validate_recipient(self, recipient: str) -> bool:
        """Validate email address"""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, recipient))


class SlackProvider(NotificationProvider):
    """Slack notification provider"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
    
    async def send(self, notification: Notification) -> bool:
        """Send Slack notification"""
        try:
            import aiohttp
            
            payload = {
                "text": notification.title,
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{notification.title}*\n{notification.message}"
                        }
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent to {notification.recipient}")
                        return True
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
                        return False
            
        except Exception as e:
            logger.error(f"Slack send failed to {notification.recipient}: {e}")
            return False
    
    async def validate_recipient(self, recipient: str) -> bool:
        """Validate Slack recipient"""
        # Slack recipients can be channels (#channel) or users (@user)
        return recipient.startswith('#') or recipient.startswith('@')


class WebhookProvider(NotificationProvider):
    """Webhook notification provider"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
    
    async def send(self, notification: Notification) -> bool:
        """Send webhook notification"""
        try:
            import aiohttp
            import hmac
            import hashlib
            
            payload = {
                "id": notification.id,
                "title": notification.title,
                "message": notification.message,
                "channel": notification.channel.value,
                "priority": notification.priority.value,
                "recipient": notification.recipient,
                "metadata": notification.metadata,
                "created_at": notification.created_at.isoformat()
            }
            
            # Create signature if secret is provided
            headers = {"Content-Type": "application/json"}
            if self.config.webhook_secret:
                payload_str = str(payload)
                signature = hmac.new(
                    self.config.webhook_secret.encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Signature"] = f"sha256={signature}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=10
                ) as response:
                    if response.status in [200, 201, 202]:
                        logger.info(f"Webhook notification sent to {notification.recipient}")
                        return True
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
                        return False
            
        except Exception as e:
            logger.error(f"Webhook send failed to {notification.recipient}: {e}")
            return False
    
    async def validate_recipient(self, recipient: str) -> bool:
        """Validate webhook recipient"""
        # Webhook recipients are typically URLs or identifiers
        return len(recipient) > 0


class NotificationService:
    """
    Advanced notification service
    
    Provides multi-channel notification capabilities with retry logic,
    scheduling, and delivery tracking.
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self._providers: Dict[NotificationChannel, NotificationProvider] = {}
        self._event_bus = get_event_bus()
        self._statistics = {
            "sent": 0,
            "failed": 0,
            "retried": 0,
            "by_channel": {channel.value: 0 for channel in NotificationChannel}
        }
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize notification providers"""
        if self.config.email_enabled:
            self._providers[NotificationChannel.EMAIL] = EmailProvider(self.config)
        
        if self.config.slack_enabled:
            self._providers[NotificationChannel.SLACK] = SlackProvider(self.config)
        
        if self.config.webhook_enabled:
            self._providers[NotificationChannel.WEBHOOK] = WebhookProvider(self.config)
    
    async def send_notification(
        self,
        title: str,
        message: str,
        channel: NotificationChannel,
        recipient: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        scheduled_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Send notification via specified channel"""
        try:
            # Create notification
            notification = Notification(
                id=f"notif_{datetime.utcnow().timestamp()}",
                title=title,
                message=message,
                channel=channel,
                priority=priority,
                recipient=recipient,
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                scheduled_at=scheduled_at,
                expires_at=expires_at
            )
            
            # Validate recipient
            provider = self._providers.get(channel)
            if not provider:
                logger.error(f"No provider available for channel {channel}")
                return False
            
            if not await provider.validate_recipient(recipient):
                logger.error(f"Invalid recipient format for {channel}: {recipient}")
                return False
            
            # Send notification
            success = await self._send_with_retry(notification)
            
            # Update statistics
            if success:
                self._statistics["sent"] += 1
                self._statistics["by_channel"][channel.value] += 1
            else:
                self._statistics["failed"] += 1
            
            # Publish event
            await self._publish_notification_event(notification, success)
            
            return success
            
        except Exception as e:
            logger.error(f"Notification send failed: {e}")
            self._statistics["failed"] += 1
            return False
    
    async def send_workflow_created_notification(
        self,
        workflow_id: str,
        name: str,
        created_at: str
    ) -> bool:
        """Send workflow created notification"""
        return await self.send_notification(
            title="Workflow Created",
            message=f"Workflow '{name}' has been created successfully.",
            channel=NotificationChannel.EMAIL,
            recipient="admin@example.com",  # This would come from user preferences
            priority=NotificationPriority.NORMAL,
            metadata={
                "workflow_id": workflow_id,
                "workflow_name": name,
                "created_at": created_at,
                "type": "workflow_created"
            }
        )
    
    async def send_workflow_deleted_notification(
        self,
        workflow_id: str,
        deleted_at: str
    ) -> bool:
        """Send workflow deleted notification"""
        return await self.send_notification(
            title="Workflow Deleted",
            message=f"Workflow {workflow_id} has been deleted.",
            channel=NotificationChannel.EMAIL,
            recipient="admin@example.com",
            priority=NotificationPriority.NORMAL,
            metadata={
                "workflow_id": workflow_id,
                "deleted_at": deleted_at,
                "type": "workflow_deleted"
            }
        )
    
    async def send_workflow_status_changed_notification(
        self,
        workflow_id: str,
        old_status: str,
        new_status: str,
        changed_at: str
    ) -> bool:
        """Send workflow status changed notification"""
        return await self.send_notification(
            title="Workflow Status Changed",
            message=f"Workflow {workflow_id} status changed from {old_status} to {new_status}.",
            channel=NotificationChannel.EMAIL,
            recipient="admin@example.com",
            priority=NotificationPriority.HIGH if new_status in ["error", "cancelled"] else NotificationPriority.NORMAL,
            metadata={
                "workflow_id": workflow_id,
                "old_status": old_status,
                "new_status": new_status,
                "changed_at": changed_at,
                "type": "workflow_status_changed"
            }
        )
    
    async def send_bulk_notifications(
        self,
        notifications: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Send multiple notifications"""
        results = {"sent": 0, "failed": 0}
        
        tasks = []
        for notif_data in notifications:
            task = asyncio.create_task(
                self.send_notification(**notif_data)
            )
            tasks.append(task)
        
        # Wait for all notifications to complete
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results_list:
            if isinstance(result, Exception):
                results["failed"] += 1
            elif result:
                results["sent"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    async def _send_with_retry(self, notification: Notification) -> bool:
        """Send notification with retry logic"""
        max_retries = notification.max_retries
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                provider = self._providers[notification.channel]
                success = await provider.send(notification)
                
                if success:
                    return True
                
                if attempt < max_retries:
                    notification.retry_count += 1
                    self._statistics["retried"] += 1
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Notification send attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
        
        return False
    
    async def _publish_notification_event(self, notification: Notification, success: bool):
        """Publish notification event"""
        event = DomainEvent(
            event_type="notification.sent" if success else "notification.failed",
            data={
                "notification_id": notification.id,
                "title": notification.title,
                "channel": notification.channel.value,
                "recipient": notification.recipient,
                "priority": notification.priority.value,
                "success": success,
                "retry_count": notification.retry_count,
                "created_at": notification.created_at.isoformat()
            },
            metadata=EventMetadata(
                source="notification_service",
                priority=3 if success else 2  # NORMAL or HIGH
            )
        )
        
        await self._event_bus.publish(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification service statistics"""
        total = self._statistics["sent"] + self._statistics["failed"]
        success_rate = (self._statistics["sent"] / total * 100) if total > 0 else 0
        
        return {
            **self._statistics,
            "success_rate": success_rate,
            "providers": list(self._providers.keys()),
            "config": {
                "email_enabled": self.config.email_enabled,
                "sms_enabled": self.config.sms_enabled,
                "push_enabled": self.config.push_enabled,
                "webhook_enabled": self.config.webhook_enabled,
                "slack_enabled": self.config.slack_enabled,
                "teams_enabled": self.config.teams_enabled,
                "discord_enabled": self.config.discord_enabled
            }
        }




