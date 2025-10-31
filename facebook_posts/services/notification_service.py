"""
Advanced Notification Service for Facebook Posts API
Real-time notifications, alerts, and communication system
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from ..core.config import get_settings
from ..core.models import FacebookPost, PostStatus, ContentType, AudienceType
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.monitoring import get_monitor, timed
from ..services.analytics_service import get_analytics_service

logger = structlog.get_logger(__name__)


class NotificationType(Enum):
    """Notification types"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"


class NotificationPriority(Enum):
    """Notification priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class NotificationStatus(Enum):
    """Notification status"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DELIVERED = "delivered"
    READ = "read"


@dataclass
class Notification:
    """Notification data structure"""
    id: str
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    recipient: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class NotificationTemplate:
    """Notification template"""
    id: str
    name: str
    type: NotificationType
    subject_template: str
    body_template: str
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationRule:
    """Notification rule for automated notifications"""
    id: str
    name: str
    condition: str
    notification_template_id: str
    recipients: List[str]
    enabled: bool = True
    cooldown_minutes: int = 60
    last_triggered: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmailNotifier:
    """Email notification service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.smtp_server = None
        self.smtp_port = None
        self.smtp_username = None
        self.smtp_password = None
        self.from_email = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize email service"""
        if not EMAIL_AVAILABLE:
            logger.warning("Email service not available - smtplib not installed")
            return
        
        try:
            self.smtp_server = self.settings.email_smtp_server
            self.smtp_port = self.settings.email_smtp_port
            self.smtp_username = self.settings.email_username
            self.smtp_password = self.settings.email_password
            self.from_email = self.settings.email_from_address
            
            if not all([self.smtp_server, self.smtp_port, self.smtp_username, self.smtp_password, self.from_email]):
                logger.warning("Email configuration incomplete")
                return
            
            self._initialized = True
            logger.info("Email service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize email service", error=str(e))
    
    @timed("email_notification")
    async def send_notification(self, notification: Notification) -> bool:
        """Send email notification"""
        if not self._initialized:
            logger.warning("Email service not initialized")
            return False
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = notification.recipient
            msg['Subject'] = notification.title
            
            # Add body
            msg.attach(MIMEText(notification.message, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info("Email notification sent", notification_id=notification.id, recipient=notification.recipient)
            return True
            
        except Exception as e:
            logger.error("Failed to send email notification", notification_id=notification.id, error=str(e))
            return False


class WebhookNotifier:
    """Webhook notification service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.timeout = 30
    
    @timed("webhook_notification")
    async def send_notification(self, notification: Notification) -> bool:
        """Send webhook notification"""
        if not REQUESTS_AVAILABLE:
            logger.warning("Webhook service not available - requests not installed")
            return False
        
        try:
            webhook_url = notification.metadata.get("webhook_url")
            if not webhook_url:
                logger.error("Webhook URL not provided", notification_id=notification.id)
                return False
            
            # Prepare payload
            payload = {
                "id": notification.id,
                "type": notification.type.value,
                "priority": notification.priority.value,
                "title": notification.title,
                "message": notification.message,
                "recipient": notification.recipient,
                "metadata": notification.metadata,
                "timestamp": notification.created_at.isoformat()
            }
            
            # Send webhook
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info("Webhook notification sent", notification_id=notification.id, webhook_url=webhook_url)
                return True
            else:
                logger.error("Webhook notification failed", notification_id=notification.id, status_code=response.status_code)
                return False
                
        except Exception as e:
            logger.error("Failed to send webhook notification", notification_id=notification.id, error=str(e))
            return False


class SlackNotifier:
    """Slack notification service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.timeout = 30
    
    @timed("slack_notification")
    async def send_notification(self, notification: Notification) -> bool:
        """Send Slack notification"""
        if not REQUESTS_AVAILABLE:
            logger.warning("Slack service not available - requests not installed")
            return False
        
        try:
            webhook_url = notification.metadata.get("slack_webhook_url")
            if not webhook_url:
                logger.error("Slack webhook URL not provided", notification_id=notification.id)
                return False
            
            # Prepare Slack payload
            color_map = {
                NotificationPriority.CRITICAL: "danger",
                NotificationPriority.HIGH: "warning",
                NotificationPriority.MEDIUM: "good",
                NotificationPriority.LOW: "#36a64f",
                NotificationPriority.INFO: "#36a64f"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(notification.priority, "#36a64f"),
                        "title": notification.title,
                        "text": notification.message,
                        "fields": [
                            {
                                "title": "Priority",
                                "value": notification.priority.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Type",
                                "value": notification.type.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Recipient",
                                "value": notification.recipient,
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": notification.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            }
                        ],
                        "footer": "Facebook Posts API",
                        "ts": int(notification.created_at.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info("Slack notification sent", notification_id=notification.id)
                return True
            else:
                logger.error("Slack notification failed", notification_id=notification.id, status_code=response.status_code)
                return False
                
        except Exception as e:
            logger.error("Failed to send Slack notification", notification_id=notification.id, error=str(e))
            return False


class NotificationTemplateManager:
    """Notification template management"""
    
    def __init__(self):
        self.templates: Dict[str, NotificationTemplate] = {}
        self.cache_manager = get_cache_manager()
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default notification templates"""
        default_templates = [
            NotificationTemplate(
                id="post_engagement_high",
                name="High Engagement Alert",
                type=NotificationType.EMAIL,
                subject_template="🚀 High Engagement Alert: {post_title}",
                body_template="""
                <h2>High Engagement Alert</h2>
                <p>Your Facebook post is performing exceptionally well!</p>
                <ul>
                    <li><strong>Post:</strong> {post_title}</li>
                    <li><strong>Engagement Rate:</strong> {engagement_rate:.1%}</li>
                    <li><strong>Views:</strong> {views:,}</li>
                    <li><strong>Likes:</strong> {likes:,}</li>
                    <li><strong>Shares:</strong> {shares:,}</li>
                    <li><strong>Comments:</strong> {comments:,}</li>
                </ul>
                <p>Keep up the great work!</p>
                """,
                variables=["post_title", "engagement_rate", "views", "likes", "shares", "comments"]
            ),
            NotificationTemplate(
                id="post_engagement_low",
                name="Low Engagement Alert",
                type=NotificationType.EMAIL,
                subject_template="⚠️ Low Engagement Alert: {post_title}",
                body_template="""
                <h2>Low Engagement Alert</h2>
                <p>Your Facebook post needs attention.</p>
                <ul>
                    <li><strong>Post:</strong> {post_title}</li>
                    <li><strong>Engagement Rate:</strong> {engagement_rate:.1%}</li>
                    <li><strong>Views:</strong> {views:,}</li>
                    <li><strong>Likes:</strong> {likes:,}</li>
                    <li><strong>Shares:</strong> {shares:,}</li>
                    <li><strong>Comments:</strong> {comments:,}</li>
                </ul>
                <p>Consider optimizing your content or posting time.</p>
                """,
                variables=["post_title", "engagement_rate", "views", "likes", "shares", "comments"]
            ),
            NotificationTemplate(
                id="system_health_warning",
                name="System Health Warning",
                type=NotificationType.SLACK,
                subject_template="⚠️ System Health Warning",
                body_template="""
                *System Health Warning*
                
                The Facebook Posts API system is experiencing issues:
                
                • *Status:* {status}
                • *Issue:* {issue_description}
                • *Timestamp:* {timestamp}
                • *Affected Components:* {affected_components}
                
                Please investigate and resolve the issue.
                """,
                variables=["status", "issue_description", "timestamp", "affected_components"]
            ),
            NotificationTemplate(
                id="ab_test_completed",
                name="A/B Test Completed",
                type=NotificationType.EMAIL,
                subject_template="📊 A/B Test Completed: {test_name}",
                body_template="""
                <h2>A/B Test Results</h2>
                <p>Your A/B test has been completed with results.</p>
                <ul>
                    <li><strong>Test Name:</strong> {test_name}</li>
                    <li><strong>Test Type:</strong> {test_type}</li>
                    <li><strong>Duration:</strong> {duration_days} days</li>
                    <li><strong>Winner:</strong> {winner_variant}</li>
                    <li><strong>Improvement:</strong> {improvement_percentage:.1f}%</li>
                    <li><strong>Statistical Significance:</strong> {significance:.1%}</li>
                </ul>
                <p>Recommendation: {recommendation}</p>
                """,
                variables=["test_name", "test_type", "duration_days", "winner_variant", "improvement_percentage", "significance", "recommendation"]
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
    
    async def get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        """Get notification template by ID"""
        return self.templates.get(template_id)
    
    async def create_template(self, template: NotificationTemplate) -> bool:
        """Create new notification template"""
        try:
            self.templates[template.id] = template
            await self.cache_manager.cache.set(f"template:{template.id}", template.__dict__, ttl=3600)
            logger.info("Notification template created", template_id=template.id)
            return True
        except Exception as e:
            logger.error("Failed to create notification template", template_id=template.id, error=str(e))
            return False
    
    async def render_template(self, template_id: str, variables: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Render notification template with variables"""
        template = await self.get_template(template_id)
        if not template:
            return None
        
        try:
            subject = template.subject_template.format(**variables)
            body = template.body_template.format(**variables)
            return subject, body
        except KeyError as e:
            logger.error("Missing template variable", template_id=template_id, missing_variable=str(e))
            return None


class NotificationRuleEngine:
    """Notification rule engine for automated notifications"""
    
    def __init__(self):
        self.rules: Dict[str, NotificationRule] = {}
        self.template_manager = NotificationTemplateManager()
        self.cache_manager = get_cache_manager()
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default notification rules"""
        default_rules = [
            NotificationRule(
                id="high_engagement_alert",
                name="High Engagement Alert",
                condition="engagement_rate > 0.8",
                notification_template_id="post_engagement_high",
                recipients=["admin@example.com"],
                cooldown_minutes=60
            ),
            NotificationRule(
                id="low_engagement_alert",
                name="Low Engagement Alert",
                condition="engagement_rate < 0.2",
                notification_template_id="post_engagement_low",
                recipients=["admin@example.com"],
                cooldown_minutes=120
            ),
            NotificationRule(
                id="system_health_warning",
                name="System Health Warning",
                condition="system_status != 'healthy'",
                notification_template_id="system_health_warning",
                recipients=["#alerts"],
                cooldown_minutes=30
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
    
    async def evaluate_rules(self, context: Dict[str, Any]) -> List[Notification]:
        """Evaluate notification rules against context"""
        notifications = []
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                time_since_last = datetime.now() - rule.last_triggered
                if time_since_last.total_seconds() < rule.cooldown_minutes * 60:
                    continue
            
            # Evaluate condition
            if self._evaluate_condition(rule.condition, context):
                # Create notification
                notification = await self._create_notification_from_rule(rule, context)
                if notification:
                    notifications.append(notification)
                    rule.last_triggered = datetime.now()
        
        return notifications
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate rule condition against context"""
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            if "engagement_rate >" in condition:
                threshold = float(condition.split(">")[1].strip())
                return context.get("engagement_rate", 0) > threshold
            elif "engagement_rate <" in condition:
                threshold = float(condition.split("<")[1].strip())
                return context.get("engagement_rate", 0) < threshold
            elif "system_status !=" in condition:
                expected_status = condition.split("!=")[1].strip().strip("'\"")
                return context.get("system_status", "") != expected_status
            else:
                return False
        except Exception as e:
            logger.error("Failed to evaluate condition", condition=condition, error=str(e))
            return False
    
    async def _create_notification_from_rule(self, rule: NotificationRule, context: Dict[str, Any]) -> Optional[Notification]:
        """Create notification from rule"""
        try:
            # Get template
            template = await self.template_manager.get_template(rule.notification_template_id)
            if not template:
                logger.error("Template not found", template_id=rule.notification_template_id)
                return None
            
            # Render template
            rendered = await self.template_manager.render_template(rule.notification_template_id, context)
            if not rendered:
                return None
            
            subject, body = rendered
            
            # Create notification for each recipient
            notifications = []
            for recipient in rule.recipients:
                notification = Notification(
                    id=f"{rule.id}_{int(time.time())}_{recipient}",
                    type=template.type,
                    priority=NotificationPriority.HIGH if "alert" in rule.name.lower() else NotificationPriority.MEDIUM,
                    title=subject,
                    message=body,
                    recipient=recipient,
                    metadata={
                        "rule_id": rule.id,
                        "template_id": rule.notification_template_id,
                        **context
                    }
                )
                notifications.append(notification)
            
            return notifications[0] if notifications else None
            
        except Exception as e:
            logger.error("Failed to create notification from rule", rule_id=rule.id, error=str(e))
            return None


class NotificationQueue:
    """Notification queue for managing notifications"""
    
    def __init__(self):
        self.queue: List[Notification] = []
        self.processing = False
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    async def add_notification(self, notification: Notification):
        """Add notification to queue"""
        self.queue.append(notification)
        await self.cache_manager.cache.set(f"notification:{notification.id}", notification.__dict__, ttl=86400)
        logger.info("Notification added to queue", notification_id=notification.id)
    
    async def process_queue(self):
        """Process notification queue"""
        if self.processing:
            return
        
        self.processing = True
        
        try:
            while self.queue:
                notification = self.queue.pop(0)
                await self._process_notification(notification)
                await asyncio.sleep(0.1)  # Small delay between notifications
        finally:
            self.processing = False
    
    async def _process_notification(self, notification: Notification):
        """Process individual notification"""
        try:
            # Get appropriate notifier
            notifier = self._get_notifier(notification.type)
            if not notifier:
                logger.error("No notifier available", notification_type=notification.type.value)
                notification.status = NotificationStatus.FAILED
                return
            
            # Send notification
            success = await notifier.send_notification(notification)
            
            if success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now()
                logger.info("Notification sent successfully", notification_id=notification.id)
            else:
                notification.status = NotificationStatus.FAILED
                notification.retry_count += 1
                
                # Retry if under max retries
                if notification.retry_count < notification.max_retries:
                    self.queue.append(notification)
                    logger.info("Notification queued for retry", notification_id=notification.id, retry_count=notification.retry_count)
                else:
                    logger.error("Notification failed after max retries", notification_id=notification.id)
            
            # Update cache
            await self.cache_manager.cache.set(f"notification:{notification.id}", notification.__dict__, ttl=86400)
            
        except Exception as e:
            logger.error("Failed to process notification", notification_id=notification.id, error=str(e))
            notification.status = NotificationStatus.FAILED
    
    def _get_notifier(self, notification_type: NotificationType):
        """Get appropriate notifier for notification type"""
        if notification_type == NotificationType.EMAIL:
            return EmailNotifier()
        elif notification_type == NotificationType.WEBHOOK:
            return WebhookNotifier()
        elif notification_type == NotificationType.SLACK:
            return SlackNotifier()
        else:
            return None


class NotificationService:
    """Main notification service orchestrator"""
    
    def __init__(self):
        self.email_notifier = EmailNotifier()
        self.webhook_notifier = WebhookNotifier()
        self.slack_notifier = SlackNotifier()
        self.template_manager = NotificationTemplateManager()
        self.rule_engine = NotificationRuleEngine()
        self.queue = NotificationQueue()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
        self.analytics_service = get_analytics_service()
        self._initialized = False
    
    async def initialize(self):
        """Initialize notification service"""
        try:
            await self.email_notifier.initialize()
            self._initialized = True
            logger.info("Notification service initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize notification service", error=str(e))
    
    @timed("notification_send")
    async def send_notification(
        self,
        notification_type: NotificationType,
        priority: NotificationPriority,
        title: str,
        message: str,
        recipient: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send notification"""
        try:
            notification = Notification(
                id=f"notif_{int(time.time())}_{hash(recipient)}",
                type=notification_type,
                priority=priority,
                title=title,
                message=message,
                recipient=recipient,
                metadata=metadata or {}
            )
            
            await self.queue.add_notification(notification)
            
            # Process queue in background
            asyncio.create_task(self.queue.process_queue())
            
            logger.info("Notification queued", notification_id=notification.id, type=notification_type.value)
            return notification.id
            
        except Exception as e:
            logger.error("Failed to send notification", error=str(e))
            raise
    
    @timed("notification_template")
    async def send_template_notification(
        self,
        template_id: str,
        recipient: str,
        variables: Dict[str, Any],
        notification_type: Optional[NotificationType] = None
    ) -> str:
        """Send notification using template"""
        try:
            # Get template
            template = await self.template_manager.get_template(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")
            
            # Render template
            rendered = await self.template_manager.render_template(template_id, variables)
            if not rendered:
                raise ValueError("Failed to render template")
            
            subject, body = rendered
            
            # Use template type if not specified
            notif_type = notification_type or template.type
            
            return await self.send_notification(
                notification_type=notif_type,
                priority=NotificationPriority.MEDIUM,
                title=subject,
                message=body,
                recipient=recipient,
                metadata={"template_id": template_id, **variables}
            )
            
        except Exception as e:
            logger.error("Failed to send template notification", template_id=template_id, error=str(e))
            raise
    
    @timed("notification_rules")
    async def evaluate_and_send_rules(self, context: Dict[str, Any]):
        """Evaluate notification rules and send notifications"""
        try:
            notifications = await self.rule_engine.evaluate_rules(context)
            
            for notification in notifications:
                await self.queue.add_notification(notification)
            
            if notifications:
                # Process queue in background
                asyncio.create_task(self.queue.process_queue())
                logger.info("Rule-based notifications queued", count=len(notifications))
            
        except Exception as e:
            logger.error("Failed to evaluate notification rules", error=str(e))
    
    async def get_notification_status(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get notification status"""
        try:
            cached_notification = await self.cache_manager.cache.get(f"notification:{notification_id}")
            if cached_notification:
                return {
                    "id": cached_notification["id"],
                    "status": cached_notification["status"],
                    "created_at": cached_notification["created_at"],
                    "sent_at": cached_notification.get("sent_at"),
                    "delivered_at": cached_notification.get("delivered_at"),
                    "read_at": cached_notification.get("read_at"),
                    "retry_count": cached_notification.get("retry_count", 0)
                }
            return None
        except Exception as e:
            logger.error("Failed to get notification status", notification_id=notification_id, error=str(e))
            return None


# Global notification service instance
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Get global notification service instance"""
    global _notification_service
    
    if _notification_service is None:
        _notification_service = NotificationService()
    
    return _notification_service


async def initialize_notification_service():
    """Initialize global notification service"""
    notification_service = get_notification_service()
    await notification_service.initialize()


# Export all classes and functions
__all__ = [
    # Enums
    'NotificationType',
    'NotificationPriority',
    'NotificationStatus',
    
    # Data classes
    'Notification',
    'NotificationTemplate',
    'NotificationRule',
    
    # Services
    'EmailNotifier',
    'WebhookNotifier',
    'SlackNotifier',
    'NotificationTemplateManager',
    'NotificationRuleEngine',
    'NotificationQueue',
    'NotificationService',
    
    # Utility functions
    'get_notification_service',
    'initialize_notification_service',
]






























