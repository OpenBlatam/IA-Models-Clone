"""
Gamma App - Notification Service
Advanced notification system with multiple channels
"""

import asyncio
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Notification types"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SLACK = "slack"
    DISCORD = "discord"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class NotificationTemplate:
    """Notification template"""
    id: str
    name: str
    type: NotificationType
    subject: str
    body: str
    variables: List[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class NotificationRecipient:
    """Notification recipient"""
    user_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    webhook_url: Optional[str] = None
    preferences: Dict[str, Any] = None

@dataclass
class Notification:
    """Notification data structure"""
    id: str
    type: NotificationType
    recipient: NotificationRecipient
    template_id: str
    subject: str
    body: str
    priority: NotificationPriority
    metadata: Dict[str, Any]
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    status: str = "pending"
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None

class NotificationService:
    """Advanced notification service"""
    
    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.templates: Dict[str, NotificationTemplate] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize rate limits
        self._init_rate_limits()
        
        # Load templates
        self._load_templates()
    
    def _init_rate_limits(self):
        """Initialize rate limits for different notification types"""
        self.rate_limits = {
            NotificationType.EMAIL: {"limit": 100, "window": 3600},  # 100 emails per hour
            NotificationType.SMS: {"limit": 50, "window": 3600},     # 50 SMS per hour
            NotificationType.PUSH: {"limit": 1000, "window": 3600},  # 1000 push per hour
            NotificationType.WEBHOOK: {"limit": 200, "window": 3600}, # 200 webhooks per hour
            NotificationType.IN_APP: {"limit": 5000, "window": 3600}, # 5000 in-app per hour
        }
    
    def _load_templates(self):
        """Load notification templates"""
        default_templates = [
            NotificationTemplate(
                id="welcome",
                name="Welcome Email",
                type=NotificationType.EMAIL,
                subject="Welcome to Gamma App!",
                body="Hello {{user_name}}, welcome to Gamma App!",
                variables=["user_name"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            NotificationTemplate(
                id="content_ready",
                name="Content Ready",
                type=NotificationType.EMAIL,
                subject="Your content is ready!",
                body="Your {{content_type}} '{{content_title}}' has been generated successfully.",
                variables=["content_type", "content_title"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            NotificationTemplate(
                id="collaboration_invite",
                name="Collaboration Invite",
                type=NotificationType.IN_APP,
                subject="Collaboration Invitation",
                body="{{inviter_name}} invited you to collaborate on '{{content_title}}'",
                variables=["inviter_name", "content_title"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            NotificationTemplate(
                id="export_complete",
                name="Export Complete",
                type=NotificationType.EMAIL,
                subject="Export Complete",
                body="Your {{format}} export of '{{content_title}}' is ready for download.",
                variables=["format", "content_title"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            NotificationTemplate(
                id="system_alert",
                name="System Alert",
                type=NotificationType.SLACK,
                subject="System Alert",
                body="🚨 {{alert_type}}: {{message}}",
                variables=["alert_type", "message"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def send_notification(
        self,
        notification: Notification,
        template_variables: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send a notification"""
        try:
            # Check rate limits
            if not self._check_rate_limit(notification.type):
                logger.warning(f"Rate limit exceeded for {notification.type}")
                return False
            
            # Render template if needed
            if notification.template_id and notification.template_id in self.templates:
                template = self.templates[notification.template_id]
                notification.subject = self._render_template(template.subject, template_variables or {})
                notification.body = self._render_template(template.body, template_variables or {})
            
            # Send based on type
            success = False
            if notification.type == NotificationType.EMAIL:
                success = await self._send_email(notification)
            elif notification.type == NotificationType.SMS:
                success = await self._send_sms(notification)
            elif notification.type == NotificationType.PUSH:
                success = await self._send_push(notification)
            elif notification.type == NotificationType.WEBHOOK:
                success = await self._send_webhook(notification)
            elif notification.type == NotificationType.IN_APP:
                success = await self._send_in_app(notification)
            elif notification.type == NotificationType.SLACK:
                success = await self._send_slack(notification)
            elif notification.type == NotificationType.DISCORD:
                success = await self._send_discord(notification)
            
            # Update notification status
            if success:
                notification.status = "sent"
                notification.sent_at = datetime.now()
            else:
                notification.status = "failed"
                notification.retry_count += 1
            
            # Store notification
            await self._store_notification(notification)
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            notification.status = "error"
            await self._store_notification(notification)
            return False
    
    async def send_bulk_notifications(
        self,
        notifications: List[Notification],
        template_variables: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Send multiple notifications"""
        results = {
            "total": len(notifications),
            "sent": 0,
            "failed": 0,
            "errors": []
        }
        
        # Process notifications in batches
        batch_size = 10
        for i in range(0, len(notifications), batch_size):
            batch = notifications[i:i + batch_size]
            
            # Process batch concurrently
            tasks = []
            for notification in batch:
                variables = template_variables.get(notification.id, {}) if template_variables else {}
                task = self.send_notification(notification, variables)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    results["failed"] += 1
                    results["errors"].append(str(result))
                elif result:
                    results["sent"] += 1
                else:
                    results["failed"] += 1
        
        return results
    
    async def schedule_notification(
        self,
        notification: Notification,
        scheduled_at: datetime,
        template_variables: Optional[Dict[str, str]] = None
    ) -> bool:
        """Schedule a notification for later delivery"""
        try:
            notification.scheduled_at = scheduled_at
            notification.status = "scheduled"
            
            # Store in Redis with TTL
            ttl = int((scheduled_at - datetime.now()).total_seconds())
            if ttl > 0:
                key = f"scheduled_notification:{notification.id}"
                data = {
                    "notification": asdict(notification),
                    "template_variables": template_variables or {}
                }
                self.redis.setex(key, ttl, json.dumps(data, default=str))
                
                # Add to scheduled set
                self.redis.zadd("scheduled_notifications", {notification.id: scheduled_at.timestamp()})
                
                return True
            else:
                # Send immediately if scheduled time has passed
                return await self.send_notification(notification, template_variables)
                
        except Exception as e:
            logger.error(f"Error scheduling notification: {e}")
            return False
    
    async def process_scheduled_notifications(self) -> int:
        """Process scheduled notifications that are due"""
        try:
            now = datetime.now()
            cutoff_time = now.timestamp()
            
            # Get due notifications
            due_notifications = self.redis.zrangebyscore(
                "scheduled_notifications", 0, cutoff_time
            )
            
            processed = 0
            for notification_id in due_notifications:
                key = f"scheduled_notification:{notification_id.decode()}"
                data = self.redis.get(key)
                
                if data:
                    notification_data = json.loads(data)
                    notification = Notification(**notification_data["notification"])
                    template_variables = notification_data.get("template_variables", {})
                    
                    # Send notification
                    success = await self.send_notification(notification, template_variables)
                    
                    if success:
                        # Remove from scheduled
                        self.redis.zrem("scheduled_notifications", notification_id)
                        self.redis.delete(key)
                        processed += 1
                    else:
                        # Retry logic
                        if notification.retry_count < notification.max_retries:
                            notification.retry_count += 1
                            # Reschedule with exponential backoff
                            delay = min(300 * (2 ** notification.retry_count), 3600)  # Max 1 hour
                            new_time = now + timedelta(seconds=delay)
                            await self.schedule_notification(notification, new_time, template_variables)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing scheduled notifications: {e}")
            return 0
    
    def _check_rate_limit(self, notification_type: NotificationType) -> bool:
        """Check if rate limit is exceeded"""
        if notification_type not in self.rate_limits:
            return True
        
        limit_config = self.rate_limits[notification_type]
        key = f"rate_limit:{notification_type.value}"
        
        # Get current count
        current_count = self.redis.get(key)
        if current_count is None:
            # First request in window
            self.redis.setex(key, limit_config["window"], 1)
            return True
        
        current_count = int(current_count)
        if current_count >= limit_config["limit"]:
            return False
        
        # Increment counter
        self.redis.incr(key)
        return True
    
    def _render_template(self, template: str, variables: Dict[str, str]) -> str:
        """Render template with variables"""
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered
    
    async def _send_email(self, notification: Notification) -> bool:
        """Send email notification"""
        try:
            if not notification.recipient.email:
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.config.get("smtp_username", "noreply@gamma.app")
            msg['To'] = notification.recipient.email
            msg['Subject'] = notification.subject
            
            msg.attach(MIMEText(notification.body, 'html'))
            
            with smtplib.SMTP(
                self.config.get("smtp_host", "localhost"),
                self.config.get("smtp_port", 587)
            ) as server:
                if self.config.get("smtp_use_tls", True):
                    server.starttls()
                
                if self.config.get("smtp_username"):
                    server.login(
                        self.config["smtp_username"],
                        self.config["smtp_password"]
                    )
                
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_sms(self, notification: Notification) -> bool:
        """Send SMS notification"""
        try:
            if not notification.recipient.phone:
                return False
            
            # This would integrate with SMS provider (Twilio, etc.)
            # For now, just log
            logger.info(f"SMS sent to {notification.recipient.phone}: {notification.body}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    async def _send_push(self, notification: Notification) -> bool:
        """Send push notification"""
        try:
            if not notification.recipient.push_token:
                return False
            
            # This would integrate with push notification service (FCM, etc.)
            # For now, just log
            logger.info(f"Push notification sent to {notification.recipient.push_token}: {notification.body}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return False
    
    async def _send_webhook(self, notification: Notification) -> bool:
        """Send webhook notification"""
        try:
            if not notification.recipient.webhook_url:
                return False
            
            payload = {
                "subject": notification.subject,
                "body": notification.body,
                "metadata": notification.metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            async with self.session.post(
                notification.recipient.webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                return response.status == 200
            
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False
    
    async def _send_in_app(self, notification: Notification) -> bool:
        """Send in-app notification"""
        try:
            # Store in Redis for real-time delivery
            key = f"in_app_notifications:{notification.recipient.user_id}"
            notification_data = {
                "id": notification.id,
                "subject": notification.subject,
                "body": notification.body,
                "type": notification.type.value,
                "priority": notification.priority.value,
                "metadata": notification.metadata,
                "created_at": datetime.now().isoformat()
            }
            
            # Add to user's notification list
            self.redis.lpush(key, json.dumps(notification_data))
            self.redis.ltrim(key, 0, 99)  # Keep only last 100 notifications
            self.redis.expire(key, 86400 * 7)  # Expire after 7 days
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending in-app notification: {e}")
            return False
    
    async def _send_slack(self, notification: Notification) -> bool:
        """Send Slack notification"""
        try:
            webhook_url = self.config.get("slack_webhook_url")
            if not webhook_url:
                return False
            
            payload = {
                "text": notification.subject,
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": notification.body
                        }
                    }
                ]
            }
            
            async with self.session.post(
                webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                return response.status == 200
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_discord(self, notification: Notification) -> bool:
        """Send Discord notification"""
        try:
            webhook_url = self.config.get("discord_webhook_url")
            if not webhook_url:
                return False
            
            payload = {
                "content": f"**{notification.subject}**\n{notification.body}",
                "embeds": [
                    {
                        "title": notification.subject,
                        "description": notification.body,
                        "color": 0x00ff00 if notification.priority == NotificationPriority.LOW else 0xff0000,
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }
            
            async with self.session.post(
                webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                return response.status == 204
            
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False
    
    async def _store_notification(self, notification: Notification):
        """Store notification in Redis"""
        try:
            key = f"notification:{notification.id}"
            data = asdict(notification)
            self.redis.setex(key, 86400 * 30, json.dumps(data, default=str))  # Store for 30 days
            
            # Add to user's notification history
            user_key = f"user_notifications:{notification.recipient.user_id}"
            self.redis.lpush(user_key, notification.id)
            self.redis.ltrim(user_key, 0, 999)  # Keep last 1000 notifications
            self.redis.expire(user_key, 86400 * 30)
            
        except Exception as e:
            logger.error(f"Error storing notification: {e}")
    
    async def get_user_notifications(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get user's notifications"""
        try:
            user_key = f"user_notifications:{user_id}"
            notification_ids = self.redis.lrange(user_key, offset, offset + limit - 1)
            
            notifications = []
            for notification_id in notification_ids:
                key = f"notification:{notification_id.decode()}"
                data = self.redis.get(key)
                if data:
                    notifications.append(json.loads(data))
            
            return notifications
            
        except Exception as e:
            logger.error(f"Error getting user notifications: {e}")
            return []
    
    async def mark_notification_read(self, user_id: str, notification_id: str) -> bool:
        """Mark notification as read"""
        try:
            key = f"read_notifications:{user_id}"
            self.redis.sadd(key, notification_id)
            self.redis.expire(key, 86400 * 30)  # Expire after 30 days
            return True
            
        except Exception as e:
            logger.error(f"Error marking notification as read: {e}")
            return False
    
    async def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        try:
            stats = {
                "total_notifications": 0,
                "sent_notifications": 0,
                "failed_notifications": 0,
                "scheduled_notifications": 0,
                "by_type": {},
                "by_priority": {}
            }
            
            # Get all notification keys
            keys = self.redis.keys("notification:*")
            stats["total_notifications"] = len(keys)
            
            # Count by status and type
            for key in keys:
                data = self.redis.get(key)
                if data:
                    notification = json.loads(data)
                    
                    # Count by status
                    if notification.get("status") == "sent":
                        stats["sent_notifications"] += 1
                    elif notification.get("status") == "failed":
                        stats["failed_notifications"] += 1
                    
                    # Count by type
                    notification_type = notification.get("type")
                    if notification_type:
                        stats["by_type"][notification_type] = stats["by_type"].get(notification_type, 0) + 1
                    
                    # Count by priority
                    priority = notification.get("priority")
                    if priority:
                        stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1
            
            # Count scheduled notifications
            scheduled_count = self.redis.zcard("scheduled_notifications")
            stats["scheduled_notifications"] = scheduled_count
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting notification stats: {e}")
            return {}
    
    def create_template(
        self,
        template_id: str,
        name: str,
        notification_type: NotificationType,
        subject: str,
        body: str,
        variables: List[str]
    ) -> bool:
        """Create a new notification template"""
        try:
            template = NotificationTemplate(
                id=template_id,
                name=name,
                type=notification_type,
                subject=subject,
                body=body,
                variables=variables,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.templates[template_id] = template
            return True
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return False
    
    def get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        """Get notification template"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[NotificationTemplate]:
        """List all notification templates"""
        return list(self.templates.values())
    
    def update_template(
        self,
        template_id: str,
        **updates
    ) -> bool:
        """Update notification template"""
        try:
            if template_id not in self.templates:
                return False
            
            template = self.templates[template_id]
            for key, value in updates.items():
                if hasattr(template, key):
                    setattr(template, key, value)
            
            template.updated_at = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error updating template: {e}")
            return False
    
    def delete_template(self, template_id: str) -> bool:
        """Delete notification template"""
        try:
            if template_id in self.templates:
                del self.templates[template_id]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting template: {e}")
            return False

























