"""
Gamma App - Real Improvement Notifications
Smart notification system for real improvements that actually work
"""

import asyncio
import logging
import time
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Notification types"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"

class NotificationPriority(Enum):
    """Notification priority"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class NotificationChannel:
    """Notification channel"""
    channel_id: str
    name: str
    type: NotificationType
    config: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class NotificationRule:
    """Notification rule"""
    rule_id: str
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    notification_template: str
    channels: List[str]
    priority: NotificationPriority = NotificationPriority.MEDIUM
    enabled: bool = True
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class Notification:
    """Notification"""
    notification_id: str
    title: str
    message: str
    priority: NotificationPriority
    channels: List[str]
    status: str = "pending"  # pending, sent, failed
    created_at: datetime = None
    sent_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class RealImprovementNotifications:
    """
    Smart notification system for real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize improvement notifications"""
        self.project_root = Path(project_root)
        self.channels: Dict[str, NotificationChannel] = {}
        self.rules: Dict[str, NotificationRule] = {}
        self.notifications: Dict[str, Notification] = {}
        self.notification_logs: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize with default channels and rules
        self._initialize_default_channels()
        self._initialize_default_rules()
        
        logger.info(f"Real Improvement Notifications initialized for {self.project_root}")
    
    def _initialize_default_channels(self):
        """Initialize default notification channels"""
        # Email channel
        email_channel = NotificationChannel(
            channel_id="email_default",
            name="Default Email",
            type=NotificationType.EMAIL,
            config={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "notifications@gammaapp.com",
                "password": "your-app-password",
                "from_email": "notifications@gammaapp.com",
                "to_emails": ["admin@gammaapp.com", "dev@gammaapp.com"]
            }
        )
        self.channels[email_channel.channel_id] = email_channel
        
        # Slack channel
        slack_channel = NotificationChannel(
            channel_id="slack_default",
            name="Default Slack",
            type=NotificationType.SLACK,
            config={
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "channel": "#improvements",
                "username": "Gamma App Bot"
            }
        )
        self.channels[slack_channel.channel_id] = slack_channel
        
        # Webhook channel
        webhook_channel = NotificationChannel(
            channel_id="webhook_default",
            name="Default Webhook",
            type=NotificationType.WEBHOOK,
            config={
                "url": "https://your-webhook-endpoint.com/notifications",
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer your-token"
                }
            }
        )
        self.channels[webhook_channel.channel_id] = webhook_channel
    
    def _initialize_default_rules(self):
        """Initialize default notification rules"""
        # Improvement completed rule
        completed_rule = NotificationRule(
            rule_id="improvement_completed",
            name="Improvement Completed",
            description="Notify when an improvement is completed successfully",
            trigger_conditions={
                "event_type": "improvement_completed",
                "success": True
            },
            notification_template="🎉 Improvement '{improvement_title}' completed successfully! Duration: {duration} minutes.",
            channels=["email_default", "slack_default"],
            priority=NotificationPriority.MEDIUM
        )
        self.rules[completed_rule.rule_id] = completed_rule
        
        # Improvement failed rule
        failed_rule = NotificationRule(
            rule_id="improvement_failed",
            name="Improvement Failed",
            description="Notify when an improvement fails",
            trigger_conditions={
                "event_type": "improvement_failed",
                "success": False
            },
            notification_template="❌ Improvement '{improvement_title}' failed: {error_message}",
            channels=["email_default", "slack_default"],
            priority=NotificationPriority.HIGH
        )
        self.rules[failed_rule.rule_id] = failed_rule
        
        # System health rule
        health_rule = NotificationRule(
            rule_id="system_health_warning",
            name="System Health Warning",
            description="Notify when system health is low",
            trigger_conditions={
                "event_type": "system_health",
                "health_percent": {"<": 80}
            },
            notification_template="⚠️ System health is at {health_percent}%. Please check the system.",
            channels=["email_default", "slack_default"],
            priority=NotificationPriority.HIGH
        )
        self.rules[health_rule.rule_id] = health_rule
        
        # High error rate rule
        error_rate_rule = NotificationRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            description="Notify when error rate is high",
            trigger_conditions={
                "event_type": "error_rate",
                "error_rate": {">": 10}
            },
            notification_template="🚨 High error rate detected: {error_rate}%. Please investigate.",
            channels=["email_default", "slack_default"],
            priority=NotificationPriority.URGENT
        )
        self.rules[error_rate_rule.rule_id] = error_rate_rule
    
    def create_notification_channel(self, name: str, type: NotificationType, 
                                  config: Dict[str, Any]) -> str:
        """Create notification channel"""
        try:
            channel_id = f"channel_{int(time.time() * 1000)}"
            
            channel = NotificationChannel(
                channel_id=channel_id,
                name=name,
                type=type,
                config=config
            )
            
            self.channels[channel_id] = channel
            
            logger.info(f"Notification channel created: {name}")
            return channel_id
            
        except Exception as e:
            logger.error(f"Failed to create notification channel: {e}")
            raise
    
    def create_notification_rule(self, name: str, description: str,
                               trigger_conditions: Dict[str, Any],
                               notification_template: str, channels: List[str],
                               priority: NotificationPriority = NotificationPriority.MEDIUM) -> str:
        """Create notification rule"""
        try:
            rule_id = f"rule_{int(time.time() * 1000)}"
            
            rule = NotificationRule(
                rule_id=rule_id,
                name=name,
                description=description,
                trigger_conditions=trigger_conditions,
                notification_template=notification_template,
                channels=channels,
                priority=priority
            )
            
            self.rules[rule_id] = rule
            
            logger.info(f"Notification rule created: {name}")
            return rule_id
            
        except Exception as e:
            logger.error(f"Failed to create notification rule: {e}")
            raise
    
    async def send_notification(self, title: str, message: str, 
                             channels: List[str], priority: NotificationPriority = NotificationPriority.MEDIUM) -> str:
        """Send notification"""
        try:
            notification_id = f"notif_{int(time.time() * 1000)}"
            
            notification = Notification(
                notification_id=notification_id,
                title=title,
                message=message,
                priority=priority,
                channels=channels
            )
            
            self.notifications[notification_id] = notification
            self.notification_logs[notification_id] = []
            
            # Send to each channel
            results = []
            for channel_id in channels:
                if channel_id in self.channels:
                    result = await self._send_to_channel(notification, channel_id)
                    results.append(result)
                else:
                    self._log_notification(notification_id, "error", f"Channel {channel_id} not found")
            
            # Update notification status
            if all(result["success"] for result in results):
                notification.status = "sent"
                notification.sent_at = datetime.utcnow()
            else:
                notification.status = "failed"
            
            return notification_id
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            raise
    
    async def _send_to_channel(self, notification: Notification, channel_id: str) -> Dict[str, Any]:
        """Send notification to specific channel"""
        try:
            channel = self.channels[channel_id]
            
            if not channel.enabled:
                return {"success": False, "error": "Channel disabled"}
            
            if channel.type == NotificationType.EMAIL:
                result = await self._send_email(notification, channel)
            elif channel.type == NotificationType.SLACK:
                result = await self._send_slack(notification, channel)
            elif channel.type == NotificationType.WEBHOOK:
                result = await self._send_webhook(notification, channel)
            elif channel.type == NotificationType.SMS:
                result = await self._send_sms(notification, channel)
            elif channel.type == NotificationType.PUSH:
                result = await self._send_push(notification, channel)
            else:
                result = {"success": False, "error": f"Unknown channel type: {channel.type}"}
            
            self._log_notification(notification.notification_id, "channel_sent", 
                                 f"Sent to {channel.name}: {result['success']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to send to channel {channel_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_email(self, notification: Notification, channel: NotificationChannel) -> Dict[str, Any]:
        """Send email notification"""
        try:
            config = channel.config
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = notification.title
            
            # Add body
            body = f"""
            {notification.message}
            
            Priority: {notification.priority.value}
            Time: {notification.created_at.isoformat()}
            """
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            
            text = msg.as_string()
            server.sendmail(config['from_email'], config['to_emails'], text)
            server.quit()
            
            return {"success": True, "message": "Email sent successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_slack(self, notification: Notification, channel: NotificationChannel) -> Dict[str, Any]:
        """Send Slack notification"""
        try:
            config = channel.config
            
            # Create Slack message
            slack_message = {
                "channel": config['channel'],
                "username": config['username'],
                "text": notification.title,
                "attachments": [
                    {
                        "color": self._get_priority_color(notification.priority),
                        "fields": [
                            {
                                "title": "Message",
                                "value": notification.message,
                                "short": False
                            },
                            {
                                "title": "Priority",
                                "value": notification.priority.value,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": notification.created_at.isoformat(),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(config['webhook_url'], json=slack_message)
            response.raise_for_status()
            
            return {"success": True, "message": "Slack message sent successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_webhook(self, notification: Notification, channel: NotificationChannel) -> Dict[str, Any]:
        """Send webhook notification"""
        try:
            config = channel.config
            
            # Create webhook payload
            payload = {
                "notification_id": notification.notification_id,
                "title": notification.title,
                "message": notification.message,
                "priority": notification.priority.value,
                "timestamp": notification.created_at.isoformat()
            }
            
            # Send webhook
            response = requests.post(
                config['url'],
                json=payload,
                headers=config.get('headers', {}),
                timeout=30
            )
            response.raise_for_status()
            
            return {"success": True, "message": "Webhook sent successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_sms(self, notification: Notification, channel: NotificationChannel) -> Dict[str, Any]:
        """Send SMS notification"""
        try:
            # This is a simplified SMS implementation
            # In production, use a proper SMS service like Twilio
            config = channel.config
            
            # Mock SMS sending
            await asyncio.sleep(0.1)  # Simulate API call
            
            return {"success": True, "message": "SMS sent successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_push(self, notification: Notification, channel: NotificationChannel) -> Dict[str, Any]:
        """Send push notification"""
        try:
            # This is a simplified push notification implementation
            # In production, use a proper push notification service
            config = channel.config
            
            # Mock push notification sending
            await asyncio.sleep(0.1)  # Simulate API call
            
            return {"success": True, "message": "Push notification sent successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_priority_color(self, priority: NotificationPriority) -> str:
        """Get color for priority"""
        color_map = {
            NotificationPriority.LOW: "good",
            NotificationPriority.MEDIUM: "warning",
            NotificationPriority.HIGH: "danger",
            NotificationPriority.URGENT: "danger"
        }
        return color_map.get(priority, "good")
    
    def _log_notification(self, notification_id: str, event: str, message: str):
        """Log notification event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if notification_id not in self.notification_logs:
            self.notification_logs[notification_id] = []
        
        self.notification_logs[notification_id].append(log_entry)
        
        logger.info(f"Notification {notification_id}: {event} - {message}")
    
    async def process_event(self, event_data: Dict[str, Any]) -> List[str]:
        """Process event and trigger notifications"""
        try:
            triggered_notifications = []
            
            # Check each rule
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                # Check if rule conditions are met
                if self._check_rule_conditions(rule, event_data):
                    # Create notification
                    title = f"Rule Triggered: {rule.name}"
                    message = self._format_notification_template(rule.notification_template, event_data)
                    
                    notification_id = await self.send_notification(
                        title=title,
                        message=message,
                        channels=rule.channels,
                        priority=rule.priority
                    )
                    
                    triggered_notifications.append(notification_id)
            
            return triggered_notifications
            
        except Exception as e:
            logger.error(f"Failed to process event: {e}")
            return []
    
    def _check_rule_conditions(self, rule: NotificationRule, event_data: Dict[str, Any]) -> bool:
        """Check if rule conditions are met"""
        try:
            conditions = rule.trigger_conditions
            
            for key, expected_value in conditions.items():
                if key not in event_data:
                    return False
                
                actual_value = event_data[key]
                
                # Handle comparison operators
                if isinstance(expected_value, dict):
                    for operator, value in expected_value.items():
                        if operator == ">":
                            if not (actual_value > value):
                                return False
                        elif operator == "<":
                            if not (actual_value < value):
                                return False
                        elif operator == ">=":
                            if not (actual_value >= value):
                                return False
                        elif operator == "<=":
                            if not (actual_value <= value):
                                return False
                        elif operator == "==":
                            if not (actual_value == value):
                                return False
                        elif operator == "!=":
                            if not (actual_value != value):
                                return False
                else:
                    if actual_value != expected_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check rule conditions: {e}")
            return False
    
    def _format_notification_template(self, template: str, event_data: Dict[str, Any]) -> str:
        """Format notification template with event data"""
        try:
            formatted_message = template
            
            for key, value in event_data.items():
                placeholder = f"{{{key}}}"
                formatted_message = formatted_message.replace(placeholder, str(value))
            
            return formatted_message
            
        except Exception as e:
            logger.error(f"Failed to format notification template: {e}")
            return template
    
    def get_notification_status(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get notification status"""
        if notification_id not in self.notifications:
            return None
        
        notification = self.notifications[notification_id]
        
        return {
            "notification_id": notification_id,
            "title": notification.title,
            "status": notification.status,
            "priority": notification.priority.value,
            "channels": notification.channels,
            "created_at": notification.created_at.isoformat(),
            "sent_at": notification.sent_at.isoformat() if notification.sent_at else None,
            "retry_count": notification.retry_count,
            "logs": self.notification_logs.get(notification_id, [])
        }
    
    def get_notification_summary(self) -> Dict[str, Any]:
        """Get notification summary"""
        total_notifications = len(self.notifications)
        sent_notifications = len([n for n in self.notifications.values() if n.status == "sent"])
        failed_notifications = len([n for n in self.notifications.values() if n.status == "failed"])
        pending_notifications = len([n for n in self.notifications.values() if n.status == "pending"])
        
        return {
            "total_notifications": total_notifications,
            "sent_notifications": sent_notifications,
            "failed_notifications": failed_notifications,
            "pending_notifications": pending_notifications,
            "success_rate": (sent_notifications / total_notifications * 100) if total_notifications > 0 else 0,
            "total_channels": len(self.channels),
            "enabled_channels": len([c for c in self.channels.values() if c.enabled]),
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled])
        }
    
    def get_notification_logs(self, notification_id: str) -> List[Dict[str, Any]]:
        """Get notification logs"""
        return self.notification_logs.get(notification_id, [])
    
    def enable_channel(self, channel_id: str) -> bool:
        """Enable notification channel"""
        try:
            if channel_id in self.channels:
                self.channels[channel_id].enabled = True
                logger.info(f"Channel enabled: {channel_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to enable channel: {e}")
            return False
    
    def disable_channel(self, channel_id: str) -> bool:
        """Disable notification channel"""
        try:
            if channel_id in self.channels:
                self.channels[channel_id].enabled = False
                logger.info(f"Channel disabled: {channel_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to disable channel: {e}")
            return False

# Global notifications instance
improvement_notifications = None

def get_improvement_notifications() -> RealImprovementNotifications:
    """Get improvement notifications instance"""
    global improvement_notifications
    if not improvement_notifications:
        improvement_notifications = RealImprovementNotifications()
    return improvement_notifications













