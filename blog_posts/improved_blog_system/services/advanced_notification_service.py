"""
Advanced Notification Service for multi-channel notifications and real-time messaging
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, or_
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import websockets
from twilio.rest import Client as TwilioClient
import firebase_admin
from firebase_admin import messaging

from ..models.database import Notification, NotificationTemplate, NotificationChannel, NotificationPreference
from ..core.exceptions import DatabaseError, ValidationError, ExternalServiceError


class NotificationType(Enum):
    """Notification types."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"


class NotificationPriority(Enum):
    """Notification priorities."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class NotificationChannel:
    """Notification channel configuration."""
    name: str
    type: NotificationType
    enabled: bool
    config: Dict[str, Any]
    rate_limit: int = 100
    priority_threshold: NotificationPriority = NotificationPriority.LOW


class AdvancedNotificationService:
    """Service for advanced notification management."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.channels = {}
        self.templates = {}
        self.websocket_connections = {}
        self._initialize_channels()
        self._initialize_templates()
    
    def _initialize_channels(self):
        """Initialize notification channels."""
        # Email channel
        self.channels["email"] = NotificationChannel(
            name="Email",
            type=NotificationType.EMAIL,
            enabled=True,
            config={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_email": "noreply@blogsystem.com"
            },
            rate_limit=1000,
            priority_threshold=NotificationPriority.LOW
        )
        
        # SMS channel
        self.channels["sms"] = NotificationChannel(
            name="SMS",
            type=NotificationType.SMS,
            enabled=True,
            config={
                "twilio_account_sid": "",
                "twilio_auth_token": "",
                "twilio_phone_number": ""
            },
            rate_limit=100,
            priority_threshold=NotificationPriority.HIGH
        )
        
        # Push notification channel
        self.channels["push"] = NotificationChannel(
            name="Push Notifications",
            type=NotificationType.PUSH,
            enabled=True,
            config={
                "firebase_credentials": "",
                "fcm_server_key": ""
            },
            rate_limit=1000,
            priority_threshold=NotificationPriority.NORMAL
        )
        
        # In-app notification channel
        self.channels["in_app"] = NotificationChannel(
            name="In-App Notifications",
            type=NotificationType.IN_APP,
            enabled=True,
            config={},
            rate_limit=10000,
            priority_threshold=NotificationPriority.LOW
        )
        
        # Webhook channel
        self.channels["webhook"] = NotificationChannel(
            name="Webhook",
            type=NotificationType.WEBHOOK,
            enabled=True,
            config={
                "webhook_url": "",
                "timeout": 30
            },
            rate_limit=500,
            priority_threshold=NotificationPriority.NORMAL
        )
        
        # Slack channel
        self.channels["slack"] = NotificationChannel(
            name="Slack",
            type=NotificationType.SLACK,
            enabled=True,
            config={
                "webhook_url": "",
                "channel": "#general"
            },
            rate_limit=100,
            priority_threshold=NotificationPriority.HIGH
        )
        
        # Discord channel
        self.channels["discord"] = NotificationChannel(
            name="Discord",
            type=NotificationType.DISCORD,
            enabled=True,
            config={
                "webhook_url": "",
                "channel_id": ""
            },
            rate_limit=100,
            priority_threshold=NotificationPriority.HIGH
        )
        
        # Telegram channel
        self.channels["telegram"] = NotificationChannel(
            name="Telegram",
            type=NotificationType.TELEGRAM,
            enabled=True,
            config={
                "bot_token": "",
                "chat_id": ""
            },
            rate_limit=30,
            priority_threshold=NotificationPriority.HIGH
        )
    
    def _initialize_templates(self):
        """Initialize notification templates."""
        self.templates = {
            "welcome": {
                "subject": "Welcome to Blog System!",
                "body": "Welcome {{user_name}}! Thank you for joining our community.",
                "channels": ["email", "in_app"]
            },
            "new_comment": {
                "subject": "New Comment on Your Post",
                "body": "{{commenter_name}} commented on your post: {{post_title}}",
                "channels": ["email", "push", "in_app"]
            },
            "post_published": {
                "subject": "Your Post Has Been Published",
                "body": "Your post '{{post_title}}' has been published successfully!",
                "channels": ["email", "in_app"]
            },
            "follow_notification": {
                "subject": "New Follower",
                "body": "{{follower_name}} started following you!",
                "channels": ["email", "push", "in_app"]
            },
            "like_notification": {
                "subject": "Someone Liked Your Post",
                "body": "{{liker_name}} liked your post: {{post_title}}",
                "channels": ["push", "in_app"]
            },
            "system_alert": {
                "subject": "System Alert",
                "body": "{{alert_message}}",
                "channels": ["email", "slack", "discord", "telegram"]
            },
            "security_alert": {
                "subject": "Security Alert",
                "body": "Security alert: {{alert_details}}",
                "channels": ["email", "sms", "slack", "discord"]
            },
            "weekly_digest": {
                "subject": "Weekly Digest",
                "body": "Here's your weekly digest with the latest posts and updates.",
                "channels": ["email"]
            }
        }
    
    async def send_notification(
        self,
        recipient: str,
        template_name: str,
        data: Dict[str, Any],
        channels: Optional[List[str]] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        scheduled_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Send notification using template."""
        try:
            if template_name not in self.templates:
                raise ValidationError(f"Template '{template_name}' not found")
            
            template = self.templates[template_name]
            
            # Use template channels if not specified
            if channels is None:
                channels = template["channels"]
            
            # Render template
            subject = self._render_template(template["subject"], data)
            body = self._render_template(template["body"], data)
            
            # Send to each channel
            results = {}
            for channel_name in channels:
                if channel_name in self.channels:
                    channel = self.channels[channel_name]
                    if channel.enabled and priority.value >= channel.priority_threshold.value:
                        result = await self._send_to_channel(
                            channel=channel,
                            recipient=recipient,
                            subject=subject,
                            body=body,
                            data=data,
                            scheduled_at=scheduled_at
                        )
                        results[channel_name] = result
            
            # Store notification record
            notification = Notification(
                recipient=recipient,
                template_name=template_name,
                subject=subject,
                body=body,
                channels=channels,
                priority=priority.value,
                data=data,
                scheduled_at=scheduled_at,
                sent_at=datetime.utcnow() if not scheduled_at else None,
                status="sent" if not scheduled_at else "scheduled"
            )
            
            self.session.add(notification)
            await self.session.commit()
            
            return {
                "success": True,
                "notification_id": notification.id,
                "results": results,
                "message": "Notification sent successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to send notification: {str(e)}")
    
    async def _send_to_channel(
        self,
        channel: NotificationChannel,
        recipient: str,
        subject: str,
        body: str,
        data: Dict[str, Any],
        scheduled_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Send notification to specific channel."""
        try:
            if channel.type == NotificationType.EMAIL:
                return await self._send_email(recipient, subject, body, channel.config)
            
            elif channel.type == NotificationType.SMS:
                return await self._send_sms(recipient, body, channel.config)
            
            elif channel.type == NotificationType.PUSH:
                return await self._send_push_notification(recipient, subject, body, channel.config)
            
            elif channel.type == NotificationType.IN_APP:
                return await self._send_in_app_notification(recipient, subject, body, data)
            
            elif channel.type == NotificationType.WEBHOOK:
                return await self._send_webhook(recipient, subject, body, channel.config)
            
            elif channel.type == NotificationType.SLACK:
                return await self._send_slack(subject, body, channel.config)
            
            elif channel.type == NotificationType.DISCORD:
                return await self._send_discord(subject, body, channel.config)
            
            elif channel.type == NotificationType.TELEGRAM:
                return await self._send_telegram(subject, body, channel.config)
            
            else:
                return {"success": False, "error": f"Unsupported channel type: {channel.type}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send email notification."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config["from_email"]
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(config["username"], config["password"])
            text = msg.as_string()
            server.sendmail(config["from_email"], recipient, text)
            server.quit()
            
            return {"success": True, "message": "Email sent successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_sms(
        self,
        recipient: str,
        body: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send SMS notification."""
        try:
            client = TwilioClient(config["twilio_account_sid"], config["twilio_auth_token"])
            
            message = client.messages.create(
                body=body,
                from_=config["twilio_phone_number"],
                to=recipient
            )
            
            return {
                "success": True,
                "message_sid": message.sid,
                "message": "SMS sent successfully"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_push_notification(
        self,
        recipient: str,
        subject: str,
        body: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send push notification."""
        try:
            # Initialize Firebase if not already done
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            
            # Create message
            message = messaging.Message(
                notification=messaging.Notification(
                    title=subject,
                    body=body
                ),
                token=recipient  # FCM token
            )
            
            # Send message
            response = messaging.send(message)
            
            return {
                "success": True,
                "message_id": response,
                "message": "Push notification sent successfully"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_in_app_notification(
        self,
        recipient: str,
        subject: str,
        body: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send in-app notification."""
        try:
            # Send via WebSocket if user is connected
            if recipient in self.websocket_connections:
                websocket = self.websocket_connections[recipient]
                notification_data = {
                    "type": "notification",
                    "subject": subject,
                    "body": body,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await websocket.send(json.dumps(notification_data))
            
            return {"success": True, "message": "In-app notification sent successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_webhook(
        self,
        recipient: str,
        subject: str,
        body: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send webhook notification."""
        try:
            webhook_data = {
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config["webhook_url"],
                    json=webhook_data,
                    timeout=aiohttp.ClientTimeout(total=config.get("timeout", 30))
                ) as response:
                    if response.status < 400:
                        return {"success": True, "message": "Webhook sent successfully"}
                    else:
                        return {"success": False, "error": f"Webhook failed with status {response.status}"}
                        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_slack(
        self,
        subject: str,
        body: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send Slack notification."""
        try:
            slack_data = {
                "channel": config["channel"],
                "text": f"*{subject}*\n{body}",
                "username": "Blog System",
                "icon_emoji": ":robot_face:"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config["webhook_url"], json=slack_data) as response:
                    if response.status < 400:
                        return {"success": True, "message": "Slack notification sent successfully"}
                    else:
                        return {"success": False, "error": f"Slack notification failed with status {response.status}"}
                        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_discord(
        self,
        subject: str,
        body: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send Discord notification."""
        try:
            discord_data = {
                "content": f"**{subject}**\n{body}",
                "username": "Blog System"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config["webhook_url"], json=discord_data) as response:
                    if response.status < 400:
                        return {"success": True, "message": "Discord notification sent successfully"}
                    else:
                        return {"success": False, "error": f"Discord notification failed with status {response.status}"}
                        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_telegram(
        self,
        subject: str,
        body: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send Telegram notification."""
        try:
            message = f"*{subject}*\n{body}"
            url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
            
            telegram_data = {
                "chat_id": config["chat_id"],
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=telegram_data) as response:
                    if response.status < 400:
                        return {"success": True, "message": "Telegram notification sent successfully"}
                    else:
                        return {"success": False, "error": f"Telegram notification failed with status {response.status}"}
                        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _render_template(self, template: str, data: Dict[str, Any]) -> str:
        """Render template with data."""
        try:
            rendered = template
            for key, value in data.items():
                placeholder = f"{{{{{key}}}}}"
                rendered = rendered.replace(placeholder, str(value))
            return rendered
        except Exception as e:
            return template
    
    async def schedule_notification(
        self,
        recipient: str,
        template_name: str,
        data: Dict[str, Any],
        scheduled_at: datetime,
        channels: Optional[List[str]] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> Dict[str, Any]:
        """Schedule notification for later delivery."""
        try:
            result = await self.send_notification(
                recipient=recipient,
                template_name=template_name,
                data=data,
                channels=channels,
                priority=priority,
                scheduled_at=scheduled_at
            )
            
            return {
                "success": True,
                "scheduled_at": scheduled_at,
                "notification_id": result["notification_id"],
                "message": "Notification scheduled successfully"
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to schedule notification: {str(e)}")
    
    async def process_scheduled_notifications(self) -> Dict[str, Any]:
        """Process scheduled notifications that are due."""
        try:
            # Get scheduled notifications that are due
            scheduled_query = select(Notification).where(
                and_(
                    Notification.status == "scheduled",
                    Notification.scheduled_at <= datetime.utcnow()
                )
            )
            
            scheduled_result = await self.session.execute(scheduled_query)
            scheduled_notifications = scheduled_result.scalars().all()
            
            processed_count = 0
            failed_count = 0
            
            for notification in scheduled_notifications:
                try:
                    # Send the notification
                    await self._send_scheduled_notification(notification)
                    processed_count += 1
                    
                except Exception as e:
                    failed_count += 1
                    # Update notification status
                    notification.status = "failed"
                    notification.error_message = str(e)
            
            await self.session.commit()
            
            return {
                "success": True,
                "processed_count": processed_count,
                "failed_count": failed_count,
                "message": f"Processed {processed_count} scheduled notifications"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to process scheduled notifications: {str(e)}")
    
    async def _send_scheduled_notification(self, notification: Notification):
        """Send a scheduled notification."""
        try:
            # Get template
            if notification.template_name not in self.templates:
                raise ValidationError(f"Template '{notification.template_name}' not found")
            
            template = self.templates[notification.template_name]
            channels = notification.channels or template["channels"]
            
            # Send to each channel
            for channel_name in channels:
                if channel_name in self.channels:
                    channel = self.channels[channel_name]
                    if channel.enabled:
                        await self._send_to_channel(
                            channel=channel,
                            recipient=notification.recipient,
                            subject=notification.subject,
                            body=notification.body,
                            data=notification.data
                        )
            
            # Update notification status
            notification.status = "sent"
            notification.sent_at = datetime.utcnow()
            
        except Exception as e:
            notification.status = "failed"
            notification.error_message = str(e)
            raise
    
    async def get_notification_history(
        self,
        recipient: Optional[str] = None,
        template_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get notification history."""
        try:
            query = select(Notification)
            
            if recipient:
                query = query.where(Notification.recipient == recipient)
            
            if template_name:
                query = query.where(Notification.template_name == template_name)
            
            if status:
                query = query.where(Notification.status == status)
            
            query = query.order_by(desc(Notification.created_at)).limit(limit).offset(offset)
            
            result = await self.session.execute(query)
            notifications = result.scalars().all()
            
            return {
                "success": True,
                "notifications": [
                    {
                        "id": n.id,
                        "recipient": n.recipient,
                        "template_name": n.template_name,
                        "subject": n.subject,
                        "body": n.body,
                        "channels": n.channels,
                        "priority": n.priority,
                        "status": n.status,
                        "scheduled_at": n.scheduled_at.isoformat() if n.scheduled_at else None,
                        "sent_at": n.sent_at.isoformat() if n.sent_at else None,
                        "created_at": n.created_at.isoformat()
                    }
                    for n in notifications
                ],
                "total": len(notifications)
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get notification history: {str(e)}")
    
    async def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        try:
            # Get total notifications
            total_query = select(func.count(Notification.id))
            total_result = await self.session.execute(total_query)
            total_notifications = total_result.scalar()
            
            # Get notifications by status
            status_query = select(
                Notification.status,
                func.count(Notification.id).label('count')
            ).group_by(Notification.status)
            
            status_result = await self.session.execute(status_query)
            status_stats = {row.status: row.count for row in status_result}
            
            # Get notifications by channel
            channel_stats = {}
            for channel_name in self.channels.keys():
                channel_query = select(func.count(Notification.id)).where(
                    Notification.channels.contains([channel_name])
                )
                channel_result = await self.session.execute(channel_query)
                channel_stats[channel_name] = channel_result.scalar()
            
            # Get notifications by template
            template_query = select(
                Notification.template_name,
                func.count(Notification.id).label('count')
            ).group_by(Notification.template_name)
            
            template_result = await self.session.execute(template_query)
            template_stats = {row.template_name: row.count for row in template_result}
            
            return {
                "total_notifications": total_notifications,
                "status_stats": status_stats,
                "channel_stats": channel_stats,
                "template_stats": template_stats,
                "available_channels": list(self.channels.keys()),
                "available_templates": list(self.templates.keys())
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get notification stats: {str(e)}")
    
    async def configure_channel(
        self,
        channel_name: str,
        config: Dict[str, Any],
        enabled: bool = True
    ) -> Dict[str, Any]:
        """Configure notification channel."""
        try:
            if channel_name not in self.channels:
                raise ValidationError(f"Channel '{channel_name}' not found")
            
            channel = self.channels[channel_name]
            channel.config.update(config)
            channel.enabled = enabled
            
            return {
                "success": True,
                "channel_name": channel_name,
                "enabled": enabled,
                "config": channel.config,
                "message": f"Channel '{channel_name}' configured successfully"
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to configure channel: {str(e)}")
    
    async def test_channel(self, channel_name: str, test_recipient: str) -> Dict[str, Any]:
        """Test notification channel."""
        try:
            if channel_name not in self.channels:
                raise ValidationError(f"Channel '{channel_name}' not found")
            
            channel = self.channels[channel_name]
            
            if not channel.enabled:
                return {
                    "success": False,
                    "error": f"Channel '{channel_name}' is disabled"
                }
            
            # Send test notification
            result = await self._send_to_channel(
                channel=channel,
                recipient=test_recipient,
                subject="Test Notification",
                body="This is a test notification from Blog System.",
                data={}
            )
            
            return {
                "success": result["success"],
                "channel_name": channel_name,
                "test_recipient": test_recipient,
                "result": result,
                "message": f"Channel '{channel_name}' test completed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_websocket_connection(self, user_id: str, websocket):
        """Add WebSocket connection for real-time notifications."""
        self.websocket_connections[user_id] = websocket
    
    async def remove_websocket_connection(self, user_id: str):
        """Remove WebSocket connection."""
        if user_id in self.websocket_connections:
            del self.websocket_connections[user_id]
    
    async def broadcast_notification(
        self,
        template_name: str,
        data: Dict[str, Any],
        channels: Optional[List[str]] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> Dict[str, Any]:
        """Broadcast notification to all connected users."""
        try:
            if template_name not in self.templates:
                raise ValidationError(f"Template '{template_name}' not found")
            
            template = self.templates[template_name]
            channels = channels or template["channels"]
            
            # Send to all connected users
            results = {}
            for user_id in self.websocket_connections.keys():
                result = await self.send_notification(
                    recipient=user_id,
                    template_name=template_name,
                    data=data,
                    channels=channels,
                    priority=priority
                )
                results[user_id] = result
            
            return {
                "success": True,
                "recipients": len(self.websocket_connections),
                "results": results,
                "message": f"Broadcast notification sent to {len(self.websocket_connections)} users"
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to broadcast notification: {str(e)}")
    
    async def get_available_templates(self) -> Dict[str, Any]:
        """Get available notification templates."""
        try:
            return {
                "templates": self.templates,
                "total": len(self.templates)
            }
        except Exception as e:
            raise DatabaseError(f"Failed to get templates: {str(e)}")
    
    async def create_custom_template(
        self,
        template_name: str,
        subject: str,
        body: str,
        channels: List[str]
    ) -> Dict[str, Any]:
        """Create custom notification template."""
        try:
            if template_name in self.templates:
                raise ValidationError(f"Template '{template_name}' already exists")
            
            # Validate channels
            for channel in channels:
                if channel not in self.channels:
                    raise ValidationError(f"Channel '{channel}' not found")
            
            self.templates[template_name] = {
                "subject": subject,
                "body": body,
                "channels": channels
            }
            
            return {
                "success": True,
                "template_name": template_name,
                "template": self.templates[template_name],
                "message": f"Template '{template_name}' created successfully"
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to create template: {str(e)}")

























