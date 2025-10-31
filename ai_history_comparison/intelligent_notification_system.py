"""
Intelligent Notification System
===============================

Advanced intelligent notification system for AI model analysis with smart
alerts, automated notifications, and intelligent routing capabilities.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
import websockets
import aiohttp
import threading
import time
import schedule
from jinja2 import Template
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications"""
    ALERT = "alert"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"
    ERROR = "error"
    CRITICAL = "critical"
    PERFORMANCE = "performance"
    ANOMALY = "anomaly"
    THRESHOLD = "threshold"
    SCHEDULED = "scheduled"
    REAL_TIME = "real_time"
    BATCH = "batch"


class NotificationChannel(str, Enum):
    """Notification channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    IN_APP = "in_app"
    DASHBOARD = "dashboard"
    LOG = "log"
    FILE = "file"
    DATABASE = "database"
    API = "api"


class NotificationPriority(str, Enum):
    """Notification priorities"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class NotificationStatus(str, Enum):
    """Notification status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NotificationRule:
    """Notification rule configuration"""
    rule_id: str
    name: str
    description: str
    condition: Dict[str, Any]
    notification_type: NotificationType
    channels: List[NotificationChannel]
    priority: NotificationPriority
    recipients: List[str]
    template: str
    cooldown_minutes: int = 15
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Notification:
    """Notification instance"""
    notification_id: str
    rule_id: str
    notification_type: NotificationType
    title: str
    message: str
    data: Dict[str, Any]
    channels: List[NotificationChannel]
    priority: NotificationPriority
    recipients: List[str]
    status: NotificationStatus
    sent_at: datetime = None
    delivered_at: datetime = None
    read_at: datetime = None
    error_message: str = ""
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class NotificationTemplate:
    """Notification template"""
    template_id: str
    name: str
    description: str
    template_type: NotificationType
    subject_template: str
    body_template: str
    variables: List[str]
    styling: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class NotificationAnalytics:
    """Notification analytics"""
    total_notifications: int
    sent_notifications: int
    delivered_notifications: int
    read_notifications: int
    failed_notifications: int
    delivery_rate: float
    read_rate: float
    average_delivery_time: float
    channel_performance: Dict[str, Dict[str, Any]]
    priority_distribution: Dict[str, int]
    type_distribution: Dict[str, int]
    time_series: Dict[str, List[Dict[str, Any]]]


class IntelligentNotificationSystem:
    """Advanced intelligent notification system for AI model analysis"""
    
    def __init__(self, max_notifications: int = 10000):
        self.max_notifications = max_notifications
        self.notification_rules: Dict[str, NotificationRule] = {}
        self.notifications: List[Notification] = []
        self.notification_templates: Dict[str, NotificationTemplate] = {}
        
        # Notification settings
        self.default_cooldown = 15  # minutes
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Channel configurations
        self.channel_configs = {
            NotificationChannel.EMAIL: {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "use_tls": True
            },
            NotificationChannel.SLACK: {
                "webhook_url": "",
                "channel": "#alerts",
                "username": "AI Analytics Bot"
            },
            NotificationChannel.TEAMS: {
                "webhook_url": "",
                "title": "AI Analytics Alert"
            },
            NotificationChannel.WEBHOOK: {
                "timeout": 30,
                "retry_attempts": 3
            }
        }
        
        # Analytics tracking
        self.analytics_data = {
            "notifications": deque(maxlen=10000),
            "delivery_times": deque(maxlen=1000),
            "channel_stats": defaultdict(lambda: {"sent": 0, "delivered": 0, "failed": 0}),
            "priority_stats": defaultdict(int),
            "type_stats": defaultdict(int)
        }
        
        # Cooldown tracking
        self.cooldown_tracker = defaultdict(lambda: defaultdict(float))
        
        # Start background tasks
        self._start_background_tasks()
    
    async def create_notification_rule(self, 
                                     name: str,
                                     description: str,
                                     condition: Dict[str, Any],
                                     notification_type: NotificationType,
                                     channels: List[NotificationChannel],
                                     priority: NotificationPriority,
                                     recipients: List[str],
                                     template: str,
                                     cooldown_minutes: int = 15) -> NotificationRule:
        """Create notification rule"""
        try:
            rule_id = hashlib.md5(f"{name}_{datetime.now()}".encode()).hexdigest()
            
            rule = NotificationRule(
                rule_id=rule_id,
                name=name,
                description=description,
                condition=condition,
                notification_type=notification_type,
                channels=channels,
                priority=priority,
                recipients=recipients,
                template=template,
                cooldown_minutes=cooldown_minutes
            )
            
            self.notification_rules[rule_id] = rule
            
            logger.info(f"Created notification rule: {name}")
            
            return rule
            
        except Exception as e:
            logger.error(f"Error creating notification rule: {str(e)}")
            raise e
    
    async def create_notification_template(self, 
                                         name: str,
                                         description: str,
                                         template_type: NotificationType,
                                         subject_template: str,
                                         body_template: str,
                                         variables: List[str],
                                         styling: Dict[str, Any] = None) -> NotificationTemplate:
        """Create notification template"""
        try:
            template_id = hashlib.md5(f"{name}_{template_type}_{datetime.now()}".encode()).hexdigest()
            
            if styling is None:
                styling = {}
            
            template = NotificationTemplate(
                template_id=template_id,
                name=name,
                description=description,
                template_type=template_type,
                subject_template=subject_template,
                body_template=body_template,
                variables=variables,
                styling=styling
            )
            
            self.notification_templates[template_id] = template
            
            logger.info(f"Created notification template: {name}")
            
            return template
            
        except Exception as e:
            logger.error(f"Error creating notification template: {str(e)}")
            raise e
    
    async def evaluate_notification_rules(self, 
                                        data: Dict[str, Any],
                                        context: Dict[str, Any] = None) -> List[Notification]:
        """Evaluate notification rules against data"""
        try:
            triggered_notifications = []
            
            if context is None:
                context = {}
            
            for rule in self.notification_rules.values():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if self._is_in_cooldown(rule.rule_id, data):
                    continue
                
                # Evaluate condition
                if await self._evaluate_condition(rule.condition, data, context):
                    # Create notification
                    notification = await self._create_notification(rule, data, context)
                    triggered_notifications.append(notification)
                    
                    # Update cooldown
                    self._update_cooldown(rule.rule_id, data)
            
            return triggered_notifications
            
        except Exception as e:
            logger.error(f"Error evaluating notification rules: {str(e)}")
            return []
    
    async def send_notification(self, 
                              notification: Notification,
                              custom_data: Dict[str, Any] = None) -> bool:
        """Send notification through specified channels"""
        try:
            success_count = 0
            total_channels = len(notification.channels)
            
            for channel in notification.channels:
                try:
                    success = await self._send_to_channel(notification, channel, custom_data)
                    if success:
                        success_count += 1
                        self._update_analytics(notification, channel, "sent")
                    else:
                        self._update_analytics(notification, channel, "failed")
                except Exception as e:
                    logger.error(f"Error sending to channel {channel}: {str(e)}")
                    self._update_analytics(notification, channel, "failed")
            
            # Update notification status
            if success_count == total_channels:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now()
            elif success_count > 0:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now()
                notification.error_message = f"Partial success: {success_count}/{total_channels} channels"
            else:
                notification.status = NotificationStatus.FAILED
                notification.error_message = "All channels failed"
            
            # Store notification
            self.notifications.append(notification)
            
            # Update analytics
            self._update_notification_analytics(notification)
            
            logger.info(f"Sent notification {notification.notification_id}: {success_count}/{total_channels} channels")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return False
    
    async def send_alert(self, 
                        title: str,
                        message: str,
                        notification_type: NotificationType = NotificationType.ALERT,
                        priority: NotificationPriority = NotificationPriority.HIGH,
                        channels: List[NotificationChannel] = None,
                        recipients: List[str] = None,
                        data: Dict[str, Any] = None) -> bool:
        """Send immediate alert"""
        try:
            if channels is None:
                channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK]
            
            if recipients is None:
                recipients = ["admin@company.com"]
            
            if data is None:
                data = {}
            
            # Create notification
            notification = Notification(
                notification_id=hashlib.md5(f"alert_{datetime.now()}".encode()).hexdigest(),
                rule_id="immediate_alert",
                notification_type=notification_type,
                title=title,
                message=message,
                data=data,
                channels=channels,
                priority=priority,
                recipients=recipients,
                status=NotificationStatus.PENDING
            )
            
            # Send notification
            success = await self.send_notification(notification)
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
            return False
    
    async def send_performance_alert(self, 
                                   model_name: str,
                                   metric: str,
                                   current_value: float,
                                   threshold: float,
                                   trend: str = "declining") -> bool:
        """Send performance alert"""
        try:
            title = f"Performance Alert: {model_name}"
            message = f"Model {model_name} {metric} is {trend}. Current value: {current_value:.3f}, Threshold: {threshold:.3f}"
            
            data = {
                "model_name": model_name,
                "metric": metric,
                "current_value": current_value,
                "threshold": threshold,
                "trend": trend,
                "alert_type": "performance"
            }
            
            success = await self.send_alert(
                title=title,
                message=message,
                notification_type=NotificationType.PERFORMANCE,
                priority=NotificationPriority.HIGH,
                data=data
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending performance alert: {str(e)}")
            return False
    
    async def send_anomaly_alert(self, 
                               model_name: str,
                               anomaly_type: str,
                               anomaly_score: float,
                               details: Dict[str, Any]) -> bool:
        """Send anomaly alert"""
        try:
            title = f"Anomaly Detected: {model_name}"
            message = f"Anomaly detected in model {model_name}. Type: {anomaly_type}, Score: {anomaly_score:.3f}"
            
            data = {
                "model_name": model_name,
                "anomaly_type": anomaly_type,
                "anomaly_score": anomaly_score,
                "details": details,
                "alert_type": "anomaly"
            }
            
            success = await self.send_alert(
                title=title,
                message=message,
                notification_type=NotificationType.ANOMALY,
                priority=NotificationPriority.URGENT,
                data=data
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending anomaly alert: {str(e)}")
            return False
    
    async def send_threshold_alert(self, 
                                 model_name: str,
                                 metric: str,
                                 current_value: float,
                                 threshold_type: str,
                                 threshold_value: float) -> bool:
        """Send threshold alert"""
        try:
            if threshold_type == "above":
                title = f"Threshold Exceeded: {model_name}"
                message = f"Model {model_name} {metric} has exceeded threshold. Current: {current_value:.3f}, Threshold: {threshold_value:.3f}"
            else:
                title = f"Threshold Below: {model_name}"
                message = f"Model {model_name} {metric} is below threshold. Current: {current_value:.3f}, Threshold: {threshold_value:.3f}"
            
            data = {
                "model_name": model_name,
                "metric": metric,
                "current_value": current_value,
                "threshold_type": threshold_type,
                "threshold_value": threshold_value,
                "alert_type": "threshold"
            }
            
            success = await self.send_alert(
                title=title,
                message=message,
                notification_type=NotificationType.THRESHOLD,
                priority=NotificationPriority.MEDIUM,
                data=data
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending threshold alert: {str(e)}")
            return False
    
    async def get_notification_analytics(self, 
                                       time_range_days: int = 30) -> NotificationAnalytics:
        """Get notification analytics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            recent_notifications = [n for n in self.notifications if n.created_at >= cutoff_date]
            
            # Calculate basic metrics
            total_notifications = len(recent_notifications)
            sent_notifications = len([n for n in recent_notifications if n.status == NotificationStatus.SENT])
            delivered_notifications = len([n for n in recent_notifications if n.status == NotificationStatus.DELIVERED])
            read_notifications = len([n for n in recent_notifications if n.status == NotificationStatus.READ])
            failed_notifications = len([n for n in recent_notifications if n.status == NotificationStatus.FAILED])
            
            # Calculate rates
            delivery_rate = delivered_notifications / sent_notifications if sent_notifications > 0 else 0
            read_rate = read_notifications / delivered_notifications if delivered_notifications > 0 else 0
            
            # Calculate average delivery time
            delivery_times = []
            for notification in recent_notifications:
                if notification.sent_at and notification.delivered_at:
                    delivery_time = (notification.delivered_at - notification.sent_at).total_seconds()
                    delivery_times.append(delivery_time)
            
            average_delivery_time = np.mean(delivery_times) if delivery_times else 0
            
            # Channel performance
            channel_performance = {}
            for channel in NotificationChannel:
                channel_stats = self.analytics_data["channel_stats"][channel.value]
                total_attempts = channel_stats["sent"] + channel_stats["failed"]
                success_rate = channel_stats["sent"] / total_attempts if total_attempts > 0 else 0
                
                channel_performance[channel.value] = {
                    "sent": channel_stats["sent"],
                    "delivered": channel_stats["delivered"],
                    "failed": channel_stats["failed"],
                    "success_rate": success_rate
                }
            
            # Priority distribution
            priority_distribution = {}
            for priority in NotificationPriority:
                count = len([n for n in recent_notifications if n.priority == priority])
                priority_distribution[priority.value] = count
            
            # Type distribution
            type_distribution = {}
            for notification_type in NotificationType:
                count = len([n for n in recent_notifications if n.notification_type == notification_type])
                type_distribution[notification_type.value] = count
            
            # Time series data
            time_series = {
                "daily_notifications": [],
                "daily_delivery_rates": [],
                "hourly_distribution": []
            }
            
            # Daily notifications
            daily_counts = defaultdict(int)
            for notification in recent_notifications:
                date_key = notification.created_at.date()
                daily_counts[date_key] += 1
            
            for date, count in sorted(daily_counts.items()):
                time_series["daily_notifications"].append({
                    "date": date.isoformat(),
                    "count": count
                })
            
            # Hourly distribution
            hourly_counts = defaultdict(int)
            for notification in recent_notifications:
                hour = notification.created_at.hour
                hourly_counts[hour] += 1
            
            for hour in range(24):
                time_series["hourly_distribution"].append({
                    "hour": hour,
                    "count": hourly_counts[hour]
                })
            
            analytics = NotificationAnalytics(
                total_notifications=total_notifications,
                sent_notifications=sent_notifications,
                delivered_notifications=delivered_notifications,
                read_notifications=read_notifications,
                failed_notifications=failed_notifications,
                delivery_rate=delivery_rate,
                read_rate=read_rate,
                average_delivery_time=average_delivery_time,
                channel_performance=channel_performance,
                priority_distribution=priority_distribution,
                type_distribution=type_distribution,
                time_series=time_series
            )
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting notification analytics: {str(e)}")
            return NotificationAnalytics(
                total_notifications=0,
                sent_notifications=0,
                delivered_notifications=0,
                read_notifications=0,
                failed_notifications=0,
                delivery_rate=0.0,
                read_rate=0.0,
                average_delivery_time=0.0,
                channel_performance={},
                priority_distribution={},
                type_distribution={},
                time_series={}
            )
    
    # Private helper methods
    async def _evaluate_condition(self, 
                                condition: Dict[str, Any], 
                                data: Dict[str, Any], 
                                context: Dict[str, Any]) -> bool:
        """Evaluate notification condition"""
        try:
            condition_type = condition.get("type", "simple")
            
            if condition_type == "simple":
                return await self._evaluate_simple_condition(condition, data, context)
            elif condition_type == "complex":
                return await self._evaluate_complex_condition(condition, data, context)
            elif condition_type == "threshold":
                return await self._evaluate_threshold_condition(condition, data, context)
            elif condition_type == "anomaly":
                return await self._evaluate_anomaly_condition(condition, data, context)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {str(e)}")
            return False
    
    async def _evaluate_simple_condition(self, 
                                       condition: Dict[str, Any], 
                                       data: Dict[str, Any], 
                                       context: Dict[str, Any]) -> bool:
        """Evaluate simple condition"""
        try:
            field = condition.get("field", "")
            operator = condition.get("operator", "equals")
            value = condition.get("value", None)
            
            if field not in data:
                return False
            
            data_value = data[field]
            
            if operator == "equals":
                return data_value == value
            elif operator == "not_equals":
                return data_value != value
            elif operator == "greater_than":
                return data_value > value
            elif operator == "less_than":
                return data_value < value
            elif operator == "greater_equal":
                return data_value >= value
            elif operator == "less_equal":
                return data_value <= value
            elif operator == "contains":
                return str(value) in str(data_value)
            elif operator == "not_contains":
                return str(value) not in str(data_value)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating simple condition: {str(e)}")
            return False
    
    async def _evaluate_complex_condition(self, 
                                        condition: Dict[str, Any], 
                                        data: Dict[str, Any], 
                                        context: Dict[str, Any]) -> bool:
        """Evaluate complex condition"""
        try:
            logic = condition.get("logic", "and")
            conditions = condition.get("conditions", [])
            
            if logic == "and":
                return all(await self._evaluate_condition(c, data, context) for c in conditions)
            elif logic == "or":
                return any(await self._evaluate_condition(c, data, context) for c in conditions)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating complex condition: {str(e)}")
            return False
    
    async def _evaluate_threshold_condition(self, 
                                          condition: Dict[str, Any], 
                                          data: Dict[str, Any], 
                                          context: Dict[str, Any]) -> bool:
        """Evaluate threshold condition"""
        try:
            field = condition.get("field", "")
            threshold = condition.get("threshold", 0)
            operator = condition.get("operator", "greater_than")
            
            if field not in data:
                return False
            
            data_value = data[field]
            
            if operator == "above":
                return data_value > threshold
            elif operator == "below":
                return data_value < threshold
            elif operator == "equals":
                return abs(data_value - threshold) < 0.001
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating threshold condition: {str(e)}")
            return False
    
    async def _evaluate_anomaly_condition(self, 
                                        condition: Dict[str, Any], 
                                        data: Dict[str, Any], 
                                        context: Dict[str, Any]) -> bool:
        """Evaluate anomaly condition"""
        try:
            anomaly_score = data.get("anomaly_score", 0)
            threshold = condition.get("threshold", 0.8)
            
            return anomaly_score > threshold
            
        except Exception as e:
            logger.error(f"Error evaluating anomaly condition: {str(e)}")
            return False
    
    async def _create_notification(self, 
                                 rule: NotificationRule, 
                                 data: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Notification:
        """Create notification from rule"""
        try:
            notification_id = hashlib.md5(f"{rule.rule_id}_{datetime.now()}".encode()).hexdigest()
            
            # Render template
            title, message = await self._render_template(rule.template, data, context)
            
            # Create notification
            notification = Notification(
                notification_id=notification_id,
                rule_id=rule.rule_id,
                notification_type=rule.notification_type,
                title=title,
                message=message,
                data=data,
                channels=rule.channels,
                priority=rule.priority,
                recipients=rule.recipients,
                status=NotificationStatus.PENDING
            )
            
            return notification
            
        except Exception as e:
            logger.error(f"Error creating notification: {str(e)}")
            raise e
    
    async def _render_template(self, 
                             template_name: str, 
                             data: Dict[str, Any], 
                             context: Dict[str, Any]) -> Tuple[str, str]:
        """Render notification template"""
        try:
            # Find template
            template = None
            for t in self.notification_templates.values():
                if t.name == template_name:
                    template = t
                    break
            
            if not template:
                # Use default template
                title = f"Alert: {data.get('model_name', 'Unknown')}"
                message = f"Notification: {data.get('message', 'No message')}"
                return title, message
            
            # Prepare template data
            template_data = {**data, **context}
            
            # Render subject
            subject_template = Template(template.subject_template)
            title = subject_template.render(**template_data)
            
            # Render body
            body_template = Template(template.body_template)
            message = body_template.render(**template_data)
            
            return title, message
            
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}")
            return "Alert", "Notification"
    
    async def _send_to_channel(self, 
                             notification: Notification, 
                             channel: NotificationChannel, 
                             custom_data: Dict[str, Any] = None) -> bool:
        """Send notification to specific channel"""
        try:
            if channel == NotificationChannel.EMAIL:
                return await self._send_email(notification, custom_data)
            elif channel == NotificationChannel.SLACK:
                return await self._send_slack(notification, custom_data)
            elif channel == NotificationChannel.TEAMS:
                return await self._send_teams(notification, custom_data)
            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook(notification, custom_data)
            elif channel == NotificationChannel.LOG:
                return await self._send_log(notification, custom_data)
            else:
                logger.warning(f"Unsupported channel: {channel}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending to channel {channel}: {str(e)}")
            return False
    
    async def _send_email(self, notification: Notification, custom_data: Dict[str, Any] = None) -> bool:
        """Send email notification"""
        try:
            config = self.channel_configs[NotificationChannel.EMAIL]
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config["username"]
            msg['To'] = ", ".join(notification.recipients)
            msg['Subject'] = notification.title
            
            # Add body
            body = notification.message
            if custom_data:
                body += f"\n\nAdditional Data: {json.dumps(custom_data, indent=2)}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            if config["use_tls"]:
                server.starttls()
            server.login(config["username"], config["password"])
            text = msg.as_string()
            server.sendmail(config["username"], notification.recipients, text)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False
    
    async def _send_slack(self, notification: Notification, custom_data: Dict[str, Any] = None) -> bool:
        """Send Slack notification"""
        try:
            config = self.channel_configs[NotificationChannel.SLACK]
            
            # Prepare payload
            payload = {
                "channel": config["channel"],
                "username": config["username"],
                "text": f"*{notification.title}*",
                "attachments": [
                    {
                        "color": self._get_slack_color(notification.notification_type),
                        "fields": [
                            {
                                "title": "Message",
                                "value": notification.message,
                                "short": False
                            }
                        ]
                    }
                ]
            }
            
            if custom_data:
                payload["attachments"][0]["fields"].append({
                    "title": "Additional Data",
                    "value": json.dumps(custom_data, indent=2),
                    "short": False
                })
            
            # Send to Slack
            response = requests.post(config["webhook_url"], json=payload, timeout=30)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return False
    
    async def _send_teams(self, notification: Notification, custom_data: Dict[str, Any] = None) -> bool:
        """Send Teams notification"""
        try:
            config = self.channel_configs[NotificationChannel.TEAMS]
            
            # Prepare payload
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": self._get_teams_color(notification.notification_type),
                "summary": notification.title,
                "sections": [
                    {
                        "activityTitle": notification.title,
                        "activitySubtitle": notification.message,
                        "facts": []
                    }
                ]
            }
            
            if custom_data:
                for key, value in custom_data.items():
                    payload["sections"][0]["facts"].append({
                        "name": key,
                        "value": str(value)
                    })
            
            # Send to Teams
            response = requests.post(config["webhook_url"], json=payload, timeout=30)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error sending Teams notification: {str(e)}")
            return False
    
    async def _send_webhook(self, notification: Notification, custom_data: Dict[str, Any] = None) -> bool:
        """Send webhook notification"""
        try:
            config = self.channel_configs[NotificationChannel.WEBHOOK]
            
            # Prepare payload
            payload = {
                "notification_id": notification.notification_id,
                "type": notification.notification_type.value,
                "title": notification.title,
                "message": notification.message,
                "priority": notification.priority.value,
                "timestamp": notification.created_at.isoformat(),
                "data": notification.data
            }
            
            if custom_data:
                payload["custom_data"] = custom_data
            
            # Send webhook (would need webhook URL configuration)
            # For now, just log
            logger.info(f"Webhook payload: {json.dumps(payload, indent=2)}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook: {str(e)}")
            return False
    
    async def _send_log(self, notification: Notification, custom_data: Dict[str, Any] = None) -> bool:
        """Send log notification"""
        try:
            log_message = f"[{notification.notification_type.value.upper()}] {notification.title}: {notification.message}"
            if custom_data:
                log_message += f" | Data: {json.dumps(custom_data)}"
            
            logger.info(log_message)
            return True
            
        except Exception as e:
            logger.error(f"Error sending log notification: {str(e)}")
            return False
    
    def _get_slack_color(self, notification_type: NotificationType) -> str:
        """Get Slack color for notification type"""
        colors = {
            NotificationType.ALERT: "warning",
            NotificationType.WARNING: "warning",
            NotificationType.INFO: "good",
            NotificationType.SUCCESS: "good",
            NotificationType.ERROR: "danger",
            NotificationType.CRITICAL: "danger",
            NotificationType.PERFORMANCE: "warning",
            NotificationType.ANOMALY: "danger",
            NotificationType.THRESHOLD: "warning"
        }
        return colors.get(notification_type, "good")
    
    def _get_teams_color(self, notification_type: NotificationType) -> str:
        """Get Teams color for notification type"""
        colors = {
            NotificationType.ALERT: "FF8C00",
            NotificationType.WARNING: "FF8C00",
            NotificationType.INFO: "0078D4",
            NotificationType.SUCCESS: "107C10",
            NotificationType.ERROR: "D13438",
            NotificationType.CRITICAL: "D13438",
            NotificationType.PERFORMANCE: "FF8C00",
            NotificationType.ANOMALY: "D13438",
            NotificationType.THRESHOLD: "FF8C00"
        }
        return colors.get(notification_type, "0078D4")
    
    def _is_in_cooldown(self, rule_id: str, data: Dict[str, Any]) -> bool:
        """Check if rule is in cooldown period"""
        try:
            # Create cooldown key based on rule and data
            cooldown_key = f"{rule_id}_{data.get('model_name', 'default')}"
            last_triggered = self.cooldown_tracker[rule_id][cooldown_key]
            
            if last_triggered == 0:
                return False
            
            # Get rule cooldown period
            rule = self.notification_rules.get(rule_id)
            if not rule:
                return False
            
            cooldown_seconds = rule.cooldown_minutes * 60
            time_since_last = time.time() - last_triggered
            
            return time_since_last < cooldown_seconds
            
        except Exception as e:
            logger.error(f"Error checking cooldown: {str(e)}")
            return False
    
    def _update_cooldown(self, rule_id: str, data: Dict[str, Any]) -> None:
        """Update cooldown tracker"""
        try:
            cooldown_key = f"{rule_id}_{data.get('model_name', 'default')}"
            self.cooldown_tracker[rule_id][cooldown_key] = time.time()
        except Exception as e:
            logger.error(f"Error updating cooldown: {str(e)}")
    
    def _update_analytics(self, notification: Notification, channel: NotificationChannel, status: str) -> None:
        """Update analytics data"""
        try:
            self.analytics_data["channel_stats"][channel.value][status] += 1
            self.analytics_data["priority_stats"][notification.priority.value] += 1
            self.analytics_data["type_stats"][notification.notification_type.value] += 1
        except Exception as e:
            logger.error(f"Error updating analytics: {str(e)}")
    
    def _update_notification_analytics(self, notification: Notification) -> None:
        """Update notification analytics"""
        try:
            self.analytics_data["notifications"].append(notification)
            
            if notification.sent_at:
                delivery_time = (notification.sent_at - notification.created_at).total_seconds()
                self.analytics_data["delivery_times"].append(delivery_time)
        except Exception as e:
            logger.error(f"Error updating notification analytics: {str(e)}")
    
    def _start_background_tasks(self) -> None:
        """Start background tasks"""
        try:
            # Start cleanup task
            cleanup_thread = threading.Thread(target=self._cleanup_old_data, daemon=True)
            cleanup_thread.start()
            
            # Start analytics task
            analytics_thread = threading.Thread(target=self._update_analytics_periodically, daemon=True)
            analytics_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {str(e)}")
    
    def _cleanup_old_data(self) -> None:
        """Cleanup old data periodically"""
        try:
            while True:
                time.sleep(3600)  # Run every hour
                
                # Cleanup old notifications
                cutoff_date = datetime.now() - timedelta(days=30)
                self.notifications = [n for n in self.notifications if n.created_at >= cutoff_date]
                
                # Cleanup old cooldown data
                current_time = time.time()
                for rule_id in list(self.cooldown_tracker.keys()):
                    for key in list(self.cooldown_tracker[rule_id].keys()):
                        if current_time - self.cooldown_tracker[rule_id][key] > 86400:  # 24 hours
                            del self.cooldown_tracker[rule_id][key]
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
    
    def _update_analytics_periodically(self) -> None:
        """Update analytics periodically"""
        try:
            while True:
                time.sleep(300)  # Run every 5 minutes
                
                # Update analytics (placeholder for future enhancements)
                pass
                
        except Exception as e:
            logger.error(f"Error in analytics task: {str(e)}")


# Global notification system instance
_notification_system: Optional[IntelligentNotificationSystem] = None


def get_intelligent_notification_system(max_notifications: int = 10000) -> IntelligentNotificationSystem:
    """Get or create global intelligent notification system instance"""
    global _notification_system
    if _notification_system is None:
        _notification_system = IntelligentNotificationSystem(max_notifications)
    return _notification_system


# Example usage
async def main():
    """Example usage of the intelligent notification system"""
    system = get_intelligent_notification_system()
    
    # Create notification template
    template = await system.create_notification_template(
        name="Performance Alert Template",
        description="Template for performance alerts",
        template_type=NotificationType.PERFORMANCE,
        subject_template="Performance Alert: {{ model_name }}",
        body_template="Model {{ model_name }} performance is {{ trend }}. Current value: {{ current_value }}, Threshold: {{ threshold }}",
        variables=["model_name", "trend", "current_value", "threshold"]
    )
    print(f"Created template: {template.template_id}")
    
    # Create notification rule
    rule = await system.create_notification_rule(
        name="Performance Threshold Rule",
        description="Alert when performance drops below threshold",
        condition={
            "type": "threshold",
            "field": "performance_score",
            "threshold": 0.8,
            "operator": "below"
        },
        notification_type=NotificationType.PERFORMANCE,
        channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
        priority=NotificationPriority.HIGH,
        recipients=["admin@company.com"],
        template="Performance Alert Template"
    )
    print(f"Created rule: {rule.rule_id}")
    
    # Send performance alert
    success = await system.send_performance_alert(
        model_name="gpt-4",
        metric="accuracy",
        current_value=0.75,
        threshold=0.8,
        trend="declining"
    )
    print(f"Performance alert sent: {success}")
    
    # Send anomaly alert
    success = await system.send_anomaly_alert(
        model_name="claude-3",
        anomaly_type="performance_drop",
        anomaly_score=0.9,
        details={"previous_score": 0.85, "current_score": 0.65}
    )
    print(f"Anomaly alert sent: {success}")
    
    # Send threshold alert
    success = await system.send_threshold_alert(
        model_name="gemini-pro",
        metric="response_time",
        current_value=5.2,
        threshold_type="above",
        threshold_value=5.0
    )
    print(f"Threshold alert sent: {success}")
    
    # Evaluate rules
    test_data = {
        "model_name": "gpt-4",
        "performance_score": 0.75,
        "anomaly_score": 0.9
    }
    
    triggered_notifications = await system.evaluate_notification_rules(test_data)
    print(f"Triggered {len(triggered_notifications)} notifications")
    
    # Get analytics
    analytics = await system.get_notification_analytics()
    print(f"Notification analytics: {analytics.total_notifications} total, {analytics.delivery_rate:.1%} delivery rate")


if __name__ == "__main__":
    asyncio.run(main())

























