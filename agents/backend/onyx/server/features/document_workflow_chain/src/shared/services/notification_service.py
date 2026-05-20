"""
Notification Service
====================

Advanced notification service for multi-channel communication.
"""

from __future__ import annotations
import asyncio
import logging
import smtplib
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import aiohttp
import requests
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioException

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution


logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Notification type enumeration"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    IN_APP = "in_app"


class NotificationPriority(str, Enum):
    """Notification priority enumeration"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(str, Enum):
    """Notification status enumeration"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NotificationTemplate:
    """Notification template representation"""
    id: str
    name: str
    type: NotificationType
    subject: str
    content: str
    variables: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=DateTimeHelpers.now_utc)
    updated_at: datetime = field(default_factory=DateTimeHelpers.now_utc)
    is_active: bool = True


@dataclass
class Notification:
    """Notification representation"""
    id: str
    type: NotificationType
    recipient: str
    subject: str
    content: str
    priority: NotificationPriority
    status: NotificationStatus
    template_id: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=DateTimeHelpers.now_utc)


@dataclass
class NotificationConfig:
    """Notification configuration"""
    # Email settings
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    
    # SMS settings (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    
    # Push notification settings
    firebase_server_key: str = ""
    apns_certificate_path: str = ""
    apns_key_path: str = ""
    
    # Webhook settings
    webhook_timeout: int = 30
    webhook_retry_count: int = 3
    
    # Slack settings
    slack_webhook_url: str = ""
    slack_bot_token: str = ""
    
    # Teams settings
    teams_webhook_url: str = ""
    
    # Discord settings
    discord_webhook_url: str = ""
    discord_bot_token: str = ""


class NotificationService:
    """Advanced notification service"""
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self.notifications: List[Notification] = []
        self.templates: Dict[str, NotificationTemplate] = {}
        self.is_running = False
        self.sender_task: Optional[asyncio.Task] = None
        self._initialize_providers()
        self._load_default_templates()
    
    def _initialize_providers(self):
        """Initialize notification providers"""
        # Initialize Twilio for SMS
        if self.config.twilio_account_sid and self.config.twilio_auth_token:
            try:
                self.twilio_client = TwilioClient(
                    self.config.twilio_account_sid,
                    self.config.twilio_auth_token
                )
                logger.info("Twilio SMS provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio: {e}")
                self.twilio_client = None
        else:
            self.twilio_client = None
        
        # Initialize HTTP session for webhooks
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout)
        )
        
        logger.info("Notification service providers initialized")
    
    def _load_default_templates(self):
        """Load default notification templates"""
        # Email templates
        self.templates["welcome_email"] = NotificationTemplate(
            id="welcome_email",
            name="Welcome Email",
            type=NotificationType.EMAIL,
            subject="Welcome to Document Workflow Chain",
            content="Hello {{name}},\n\nWelcome to Document Workflow Chain! We're excited to have you on board.\n\nBest regards,\nThe Team",
            variables=["name"]
        )
        
        self.templates["workflow_completed"] = NotificationTemplate(
            id="workflow_completed",
            name="Workflow Completed",
            type=NotificationType.EMAIL,
            subject="Workflow {{workflow_name}} Completed",
            content="Hello {{user_name}},\n\nYour workflow '{{workflow_name}}' has been completed successfully.\n\nWorkflow ID: {{workflow_id}}\nCompleted at: {{completed_at}}\n\nBest regards,\nThe System",
            variables=["user_name", "workflow_name", "workflow_id", "completed_at"]
        )
        
        # SMS templates
        self.templates["otp_sms"] = NotificationTemplate(
            id="otp_sms",
            name="OTP SMS",
            type=NotificationType.SMS,
            subject="",
            content="Your OTP code is: {{otp_code}}. Valid for 5 minutes.",
            variables=["otp_code"]
        )
        
        # Push notification templates
        self.templates["workflow_alert"] = NotificationTemplate(
            id="workflow_alert",
            name="Workflow Alert",
            type=NotificationType.PUSH,
            subject="Workflow Alert",
            content="Workflow {{workflow_name}} requires your attention.",
            variables=["workflow_name"]
        )
        
        logger.info(f"Loaded {len(self.templates)} default templates")
    
    async def start(self):
        """Start the notification service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.sender_task = asyncio.create_task(self._sender_worker())
        logger.info("Notification service started")
    
    async def stop(self):
        """Stop the notification service"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.sender_task:
            self.sender_task.cancel()
            try:
                await self.sender_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        await self.http_session.close()
        
        logger.info("Notification service stopped")
    
    def create_template(
        self,
        name: str,
        type: NotificationType,
        subject: str,
        content: str,
        variables: Optional[List[str]] = None
    ) -> str:
        """Create notification template"""
        template_id = f"{name.lower().replace(' ', '_')}_{int(DateTimeHelpers.now_utc().timestamp())}"
        
        template = NotificationTemplate(
            id=template_id,
            name=name,
            type=type,
            subject=subject,
            content=content,
            variables=variables or []
        )
        
        self.templates[template_id] = template
        logger.info(f"Created notification template: {template_id}")
        
        return template_id
    
    def send_notification(
        self,
        type: NotificationType,
        recipient: str,
        subject: str,
        content: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        template_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scheduled_at: Optional[datetime] = None
    ) -> str:
        """Send notification"""
        notification_id = f"notif_{int(DateTimeHelpers.now_utc().timestamp())}_{len(self.notifications)}"
        
        notification = Notification(
            id=notification_id,
            type=type,
            recipient=recipient,
            subject=subject,
            content=content,
            priority=priority,
            status=NotificationStatus.PENDING,
            template_id=template_id,
            variables=variables or {},
            metadata=metadata or {},
            scheduled_at=scheduled_at
        )
        
        self.notifications.append(notification)
        logger.info(f"Created notification: {notification_id}")
        
        return notification_id
    
    def send_template_notification(
        self,
        template_id: str,
        recipient: str,
        variables: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        scheduled_at: Optional[datetime] = None
    ) -> str:
        """Send notification using template"""
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
        
        template = self.templates[template_id]
        
        # Replace variables in subject and content
        subject = self._replace_variables(template.subject, variables)
        content = self._replace_variables(template.content, variables)
        
        return self.send_notification(
            type=template.type,
            recipient=recipient,
            subject=subject,
            content=content,
            priority=priority,
            template_id=template_id,
            variables=variables,
            metadata=metadata,
            scheduled_at=scheduled_at
        )
    
    def _replace_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Replace variables in text"""
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            text = text.replace(placeholder, str(value))
        return text
    
    async def _sender_worker(self):
        """Notification sender worker"""
        while self.is_running:
            try:
                await self._process_pending_notifications()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Notification sender error: {e}")
                await asyncio.sleep(5)
    
    async def _process_pending_notifications(self):
        """Process pending notifications"""
        now = DateTimeHelpers.now_utc()
        
        # Get pending notifications that are ready to send
        pending_notifications = [
            n for n in self.notifications
            if n.status == NotificationStatus.PENDING and
            (n.scheduled_at is None or n.scheduled_at <= now)
        ]
        
        # Sort by priority
        priority_order = {
            NotificationPriority.URGENT: 0,
            NotificationPriority.HIGH: 1,
            NotificationPriority.NORMAL: 2,
            NotificationPriority.LOW: 3
        }
        pending_notifications.sort(key=lambda x: priority_order[x.priority])
        
        # Process notifications
        for notification in pending_notifications:
            try:
                await self._send_notification(notification)
            except Exception as e:
                logger.error(f"Failed to send notification {notification.id}: {e}")
                await self._handle_notification_failure(notification, str(e))
    
    async def _send_notification(self, notification: Notification):
        """Send individual notification"""
        notification.status = NotificationStatus.SENT
        notification.sent_at = DateTimeHelpers.now_utc()
        
        try:
            if notification.type == NotificationType.EMAIL:
                await self._send_email(notification)
            elif notification.type == NotificationType.SMS:
                await self._send_sms(notification)
            elif notification.type == NotificationType.PUSH:
                await self._send_push(notification)
            elif notification.type == NotificationType.WEBHOOK:
                await self._send_webhook(notification)
            elif notification.type == NotificationType.SLACK:
                await self._send_slack(notification)
            elif notification.type == NotificationType.TEAMS:
                await self._send_teams(notification)
            elif notification.type == NotificationType.DISCORD:
                await self._send_discord(notification)
            elif notification.type == NotificationType.IN_APP:
                await self._send_in_app(notification)
            else:
                raise ValueError(f"Unsupported notification type: {notification.type}")
            
            notification.status = NotificationStatus.DELIVERED
            notification.delivered_at = DateTimeHelpers.now_utc()
            
            logger.info(f"Notification {notification.id} delivered successfully")
        
        except Exception as e:
            await self._handle_notification_failure(notification, str(e))
    
    async def _send_email(self, notification: Notification):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = notification.recipient
            msg['Subject'] = notification.subject
            
            msg.attach(MIMEText(notification.content, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
            if self.config.smtp_use_tls:
                server.starttls()
            
            if self.config.smtp_username and self.config.smtp_password:
                server.login(self.config.smtp_username, self.config.smtp_password)
            
            text = msg.as_string()
            server.sendmail(self.config.smtp_username, notification.recipient, text)
            server.quit()
            
        except Exception as e:
            raise Exception(f"Email sending failed: {e}")
    
    async def _send_sms(self, notification: Notification):
        """Send SMS notification"""
        if not self.twilio_client:
            raise Exception("Twilio client not initialized")
        
        try:
            message = self.twilio_client.messages.create(
                body=notification.content,
                from_=self.config.twilio_phone_number,
                to=notification.recipient
            )
            
            logger.info(f"SMS sent successfully: {message.sid}")
        
        except TwilioException as e:
            raise Exception(f"SMS sending failed: {e}")
    
    async def _send_push(self, notification: Notification):
        """Send push notification"""
        # Simplified push notification implementation
        # In a real implementation, you would use Firebase, APNS, etc.
        logger.info(f"Push notification sent to {notification.recipient}: {notification.content}")
    
    async def _send_webhook(self, notification: Notification):
        """Send webhook notification"""
        webhook_url = notification.metadata.get("webhook_url")
        if not webhook_url:
            raise Exception("Webhook URL not provided")
        
        payload = {
            "subject": notification.subject,
            "content": notification.content,
            "recipient": notification.recipient,
            "metadata": notification.metadata
        }
        
        try:
            async with self.http_session.post(webhook_url, json=payload) as response:
                if response.status >= 400:
                    raise Exception(f"Webhook returned status {response.status}")
        
        except Exception as e:
            raise Exception(f"Webhook sending failed: {e}")
    
    async def _send_slack(self, notification: Notification):
        """Send Slack notification"""
        webhook_url = self.config.slack_webhook_url
        if not webhook_url:
            raise Exception("Slack webhook URL not configured")
        
        payload = {
            "text": f"*{notification.subject}*\n{notification.content}",
            "channel": notification.metadata.get("channel", "#general"),
            "username": "Workflow Bot"
        }
        
        try:
            async with self.http_session.post(webhook_url, json=payload) as response:
                if response.status >= 400:
                    raise Exception(f"Slack webhook returned status {response.status}")
        
        except Exception as e:
            raise Exception(f"Slack sending failed: {e}")
    
    async def _send_teams(self, notification: Notification):
        """Send Microsoft Teams notification"""
        webhook_url = self.config.teams_webhook_url
        if not webhook_url:
            raise Exception("Teams webhook URL not configured")
        
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "0076D7",
            "summary": notification.subject,
            "sections": [{
                "activityTitle": notification.subject,
                "activitySubtitle": notification.content,
                "markdown": True
            }]
        }
        
        try:
            async with self.http_session.post(webhook_url, json=payload) as response:
                if response.status >= 400:
                    raise Exception(f"Teams webhook returned status {response.status}")
        
        except Exception as e:
            raise Exception(f"Teams sending failed: {e}")
    
    async def _send_discord(self, notification: Notification):
        """Send Discord notification"""
        webhook_url = self.config.discord_webhook_url
        if not webhook_url:
            raise Exception("Discord webhook URL not configured")
        
        payload = {
            "content": f"**{notification.subject}**\n{notification.content}",
            "username": "Workflow Bot"
        }
        
        try:
            async with self.http_session.post(webhook_url, json=payload) as response:
                if response.status >= 400:
                    raise Exception(f"Discord webhook returned status {response.status}")
        
        except Exception as e:
            raise Exception(f"Discord sending failed: {e}")
    
    async def _send_in_app(self, notification: Notification):
        """Send in-app notification"""
        # Simplified in-app notification implementation
        # In a real implementation, you would store in database and notify via WebSocket
        logger.info(f"In-app notification sent to {notification.recipient}: {notification.content}")
    
    async def _handle_notification_failure(self, notification: Notification, error_message: str):
        """Handle notification failure"""
        notification.retry_count += 1
        notification.error_message = error_message
        
        if notification.retry_count >= notification.max_retries:
            notification.status = NotificationStatus.FAILED
            notification.failed_at = DateTimeHelpers.now_utc()
            logger.error(f"Notification {notification.id} failed after {notification.max_retries} retries")
        else:
            notification.status = NotificationStatus.PENDING
            # Schedule retry with exponential backoff
            retry_delay = 2 ** notification.retry_count  # 2, 4, 8 seconds
            notification.scheduled_at = DateTimeHelpers.now_utc() + timedelta(seconds=retry_delay)
            logger.warning(f"Notification {notification.id} failed, retrying in {retry_delay} seconds")
    
    def get_notification_status(self, notification_id: str) -> Optional[Notification]:
        """Get notification status"""
        for notification in self.notifications:
            if notification.id == notification_id:
                return notification
        return None
    
    def get_notifications_by_recipient(self, recipient: str, limit: int = 100) -> List[Notification]:
        """Get notifications by recipient"""
        notifications = [n for n in self.notifications if n.recipient == recipient]
        return notifications[-limit:]
    
    def get_notifications_by_type(self, type: NotificationType, limit: int = 100) -> List[Notification]:
        """Get notifications by type"""
        notifications = [n for n in self.notifications if n.type == type]
        return notifications[-limit:]
    
    def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        total_notifications = len(self.notifications)
        
        # Status distribution
        statuses = {}
        for notification in self.notifications:
            statuses[notification.status.value] = statuses.get(notification.status.value, 0) + 1
        
        # Type distribution
        types = {}
        for notification in self.notifications:
            types[notification.type.value] = types.get(notification.type.value, 0) + 1
        
        # Priority distribution
        priorities = {}
        for notification in self.notifications:
            priorities[notification.priority.value] = priorities.get(notification.priority.value, 0) + 1
        
        # Recent activity (last 24 hours)
        recent_cutoff = DateTimeHelpers.now_utc() - timedelta(hours=24)
        recent_notifications = [n for n in self.notifications if n.created_at > recent_cutoff]
        
        return {
            "total_notifications": total_notifications,
            "recent_notifications_24h": len(recent_notifications),
            "statuses": statuses,
            "types": types,
            "priorities": priorities,
            "templates_count": len(self.templates),
            "timestamp": DateTimeHelpers.now_utc().isoformat()
        }
    
    def cancel_notification(self, notification_id: str) -> bool:
        """Cancel notification"""
        for notification in self.notifications:
            if notification.id == notification_id and notification.status == NotificationStatus.PENDING:
                notification.status = NotificationStatus.CANCELLED
                logger.info(f"Notification {notification_id} cancelled")
                return True
        return False
    
    def resend_notification(self, notification_id: str) -> bool:
        """Resend notification"""
        for notification in self.notifications:
            if notification.id == notification_id and notification.status in [NotificationStatus.FAILED, NotificationStatus.CANCELLED]:
                notification.status = NotificationStatus.PENDING
                notification.retry_count = 0
                notification.error_message = None
                notification.scheduled_at = None
                logger.info(f"Notification {notification_id} queued for resending")
                return True
        return False


# Global notification service
notification_service = NotificationService()


# Utility functions
async def start_notification_service():
    """Start the notification service"""
    await notification_service.start()


async def stop_notification_service():
    """Stop the notification service"""
    await notification_service.stop()


def send_notification(
    type: NotificationType,
    recipient: str,
    subject: str,
    content: str,
    priority: NotificationPriority = NotificationPriority.NORMAL,
    template_id: Optional[str] = None,
    variables: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    scheduled_at: Optional[datetime] = None
) -> str:
    """Send notification"""
    return notification_service.send_notification(
        type, recipient, subject, content, priority,
        template_id, variables, metadata, scheduled_at
    )


def send_template_notification(
    template_id: str,
    recipient: str,
    variables: Dict[str, Any],
    priority: NotificationPriority = NotificationPriority.NORMAL,
    metadata: Optional[Dict[str, Any]] = None,
    scheduled_at: Optional[datetime] = None
) -> str:
    """Send template notification"""
    return notification_service.send_template_notification(
        template_id, recipient, variables, priority, metadata, scheduled_at
    )


def get_notification_status(notification_id: str) -> Optional[Notification]:
    """Get notification status"""
    return notification_service.get_notification_status(notification_id)


def get_notification_statistics() -> Dict[str, Any]:
    """Get notification statistics"""
    return notification_service.get_notification_statistics()


# Common notification functions
def send_welcome_email(user_email: str, user_name: str):
    """Send welcome email"""
    return send_template_notification(
        "welcome_email",
        user_email,
        {"name": user_name}
    )


def send_workflow_completed_notification(user_email: str, user_name: str, workflow_name: str, workflow_id: str):
    """Send workflow completed notification"""
    return send_template_notification(
        "workflow_completed",
        user_email,
        {
            "user_name": user_name,
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "completed_at": DateTimeHelpers.now_utc().isoformat()
        }
    )


def send_otp_sms(phone_number: str, otp_code: str):
    """Send OTP SMS"""
    return send_template_notification(
        "otp_sms",
        phone_number,
        {"otp_code": otp_code},
        NotificationPriority.HIGH
    )


def send_workflow_alert_push(user_id: str, workflow_name: str):
    """Send workflow alert push notification"""
    return send_template_notification(
        "workflow_alert",
        user_id,
        {"workflow_name": workflow_name},
        NotificationPriority.HIGH
    )




