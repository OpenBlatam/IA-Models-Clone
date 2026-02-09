"""
Notification Service
===================

Advanced notification system for the Bulk TruthGPT system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import aiohttp
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class NotificationType(str, Enum):
    """Notification types."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"

class NotificationPriority(str, Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class Notification:
    """Notification data structure."""
    id: str
    type: NotificationType
    recipient: str
    subject: str
    message: str
    priority: NotificationPriority
    metadata: Dict[str, Any]
    created_at: datetime
    sent_at: Optional[datetime] = None
    status: str = "pending"

class NotificationService:
    """
    Advanced notification service.
    
    Features:
    - Multiple notification channels
    - Priority-based delivery
    - Retry mechanism
    - Template support
    - Rate limiting
    - Delivery tracking
    """
    
    def __init__(self):
        self.notifications = []
        self.templates = {}
        self.rate_limits = {}
        self.retry_queue = []
        
    async def initialize(self):
        """Initialize notification service."""
        logger.info("Initializing Notification Service...")
        
        try:
            # Load notification templates
            await self._load_templates()
            
            # Start background tasks
            asyncio.create_task(self._process_notifications())
            asyncio.create_task(self._process_retry_queue())
            
            logger.info("Notification Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Notification Service: {str(e)}")
            raise
    
    async def _load_templates(self):
        """Load notification templates."""
        try:
            # Default templates
            self.templates = {
                'task_started': {
                    'subject': 'Task Started: {task_id}',
                    'message': 'Your bulk generation task has started.\nTask ID: {task_id}\nQuery: {query}\nMax Documents: {max_documents}'
                },
                'task_completed': {
                    'subject': 'Task Completed: {task_id}',
                    'message': 'Your bulk generation task has completed.\nTask ID: {task_id}\nDocuments Generated: {documents_generated}\nDuration: {duration}'
                },
                'task_failed': {
                    'subject': 'Task Failed: {task_id}',
                    'message': 'Your bulk generation task has failed.\nTask ID: {task_id}\nError: {error}\nPlease check the logs for more details.'
                },
                'system_alert': {
                    'subject': 'System Alert: {alert_type}',
                    'message': 'System alert: {alert_type}\nMessage: {message}\nTimestamp: {timestamp}'
                },
                'quality_alert': {
                    'subject': 'Quality Alert: {task_id}',
                    'message': 'Quality alert for task {task_id}.\nQuality Score: {quality_score}\nThreshold: {threshold}\nThis may indicate issues with content generation.'
                }
            }
            
            logger.info("Notification templates loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load templates: {str(e)}")
    
    async def send_notification(
        self,
        notification_type: NotificationType,
        recipient: str,
        subject: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        template: Optional[str] = None,
        template_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send notification.
        
        Args:
            notification_type: Type of notification
            recipient: Recipient address/ID
            subject: Notification subject
            message: Notification message
            priority: Notification priority
            metadata: Additional metadata
            template: Template name to use
            template_data: Data for template rendering
            
        Returns:
            Notification ID
        """
        try:
            # Generate notification ID
            import uuid
            notification_id = str(uuid.uuid4())
            
            # Use template if provided
            if template and template in self.templates:
                template_info = self.templates[template]
                if template_data:
                    subject = template_info['subject'].format(**template_data)
                    message = template_info['message'].format(**template_data)
                else:
                    subject = template_info['subject']
                    message = template_info['message']
            
            # Create notification
            notification = Notification(
                id=notification_id,
                type=notification_type,
                recipient=recipient,
                subject=subject,
                message=message,
                priority=priority,
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            
            # Add to notifications list
            self.notifications.append(notification)
            
            # Check rate limits
            if await self._check_rate_limit(notification_type, recipient):
                logger.info(f"Notification {notification_id} queued for delivery")
                return notification_id
            else:
                logger.warning(f"Rate limit exceeded for {notification_type} to {recipient}")
                return notification_id
                
        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")
            raise
    
    async def _check_rate_limit(self, notification_type: NotificationType, recipient: str) -> bool:
        """Check rate limits for notification."""
        try:
            key = f"{notification_type.value}:{recipient}"
            current_time = datetime.utcnow()
            
            # Rate limits (per hour)
            limits = {
                NotificationType.EMAIL: 10,
                NotificationType.WEBHOOK: 100,
                NotificationType.SLACK: 20,
                NotificationType.DISCORD: 20,
                NotificationType.TELEGRAM: 20
            }
            
            limit = limits.get(notification_type, 10)
            
            # Count recent notifications
            if key not in self.rate_limits:
                self.rate_limits[key] = []
            
            # Remove old entries (older than 1 hour)
            cutoff_time = current_time.timestamp() - 3600
            self.rate_limits[key] = [
                timestamp for timestamp in self.rate_limits[key]
                if timestamp > cutoff_time
            ]
            
            # Check if under limit
            if len(self.rate_limits[key]) < limit:
                self.rate_limits[key].append(current_time.timestamp())
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to check rate limit: {str(e)}")
            return True  # Allow if check fails
    
    async def _process_notifications(self):
        """Process pending notifications."""
        while True:
            try:
                await asyncio.sleep(5)  # Process every 5 seconds
                
                # Get pending notifications
                pending_notifications = [
                    n for n in self.notifications
                    if n.status == "pending"
                ]
                
                # Sort by priority and creation time
                priority_order = {
                    NotificationPriority.URGENT: 0,
                    NotificationPriority.HIGH: 1,
                    NotificationPriority.NORMAL: 2,
                    NotificationPriority.LOW: 3
                }
                
                pending_notifications.sort(
                    key=lambda n: (priority_order[n.priority], n.created_at)
                )
                
                # Process notifications (max 10 at a time)
                for notification in pending_notifications[:10]:
                    try:
                        await self._deliver_notification(notification)
                    except Exception as e:
                        logger.error(f"Failed to deliver notification {notification.id}: {str(e)}")
                        # Add to retry queue
                        self.retry_queue.append({
                            'notification': notification,
                            'retry_count': 0,
                            'next_retry': datetime.utcnow().timestamp() + 300  # 5 minutes
                        })
                
            except Exception as e:
                logger.error(f"Error processing notifications: {str(e)}")
    
    async def _deliver_notification(self, notification: Notification):
        """Deliver notification."""
        try:
            if notification.type == NotificationType.EMAIL:
                await self._send_email(notification)
            elif notification.type == NotificationType.WEBHOOK:
                await self._send_webhook(notification)
            elif notification.type == NotificationType.SLACK:
                await self._send_slack(notification)
            elif notification.type == NotificationType.DISCORD:
                await self._send_discord(notification)
            elif notification.type == NotificationType.TELEGRAM:
                await self._send_telegram(notification)
            else:
                logger.warning(f"Unknown notification type: {notification.type}")
                return
            
            # Update notification status
            notification.status = "sent"
            notification.sent_at = datetime.utcnow()
            
            logger.info(f"Notification {notification.id} delivered successfully")
            
        except Exception as e:
            logger.error(f"Failed to deliver notification {notification.id}: {str(e)}")
            notification.status = "failed"
            raise
    
    async def _send_email(self, notification: Notification):
        """Send email notification."""
        try:
            # This would integrate with your email service
            # For now, just log the email
            logger.info(f"Email sent to {notification.recipient}: {notification.subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            raise
    
    async def _send_webhook(self, notification: Notification):
        """Send webhook notification."""
        try:
            webhook_url = notification.recipient
            
            payload = {
                'subject': notification.subject,
                'message': notification.message,
                'metadata': notification.metadata,
                'timestamp': notification.created_at.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=10) as response:
                    if response.status >= 400:
                        raise Exception(f"Webhook failed with status {response.status}")
            
            logger.info(f"Webhook sent to {webhook_url}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook: {str(e)}")
            raise
    
    async def _send_slack(self, notification: Notification):
        """Send Slack notification."""
        try:
            # This would integrate with Slack API
            logger.info(f"Slack message sent to {notification.recipient}: {notification.subject}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {str(e)}")
            raise
    
    async def _send_discord(self, notification: Notification):
        """Send Discord notification."""
        try:
            # This would integrate with Discord API
            logger.info(f"Discord message sent to {notification.recipient}: {notification.subject}")
            
        except Exception as e:
            logger.error(f"Failed to send Discord message: {str(e)}")
            raise
    
    async def _send_telegram(self, notification: Notification):
        """Send Telegram notification."""
        try:
            # This would integrate with Telegram Bot API
            logger.info(f"Telegram message sent to {notification.recipient}: {notification.subject}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {str(e)}")
            raise
    
    async def _process_retry_queue(self):
        """Process retry queue for failed notifications."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow().timestamp()
                
                # Process retries
                for retry_item in self.retry_queue[:]:
                    if current_time >= retry_item['next_retry']:
                        try:
                            await self._deliver_notification(retry_item['notification'])
                            self.retry_queue.remove(retry_item)
                        except Exception as e:
                            retry_item['retry_count'] += 1
                            
                            if retry_item['retry_count'] >= 3:
                                logger.error(f"Notification {retry_item['notification'].id} failed after 3 retries")
                                retry_item['notification'].status = "failed"
                                self.retry_queue.remove(retry_item)
                            else:
                                # Exponential backoff
                                delay = 300 * (2 ** retry_item['retry_count'])
                                retry_item['next_retry'] = current_time + delay
                
            except Exception as e:
                logger.error(f"Error processing retry queue: {str(e)}")
    
    async def get_notification_status(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get notification status."""
        try:
            for notification in self.notifications:
                if notification.id == notification_id:
                    return {
                        'id': notification.id,
                        'type': notification.type.value,
                        'recipient': notification.recipient,
                        'subject': notification.subject,
                        'status': notification.status,
                        'created_at': notification.created_at.isoformat(),
                        'sent_at': notification.sent_at.isoformat() if notification.sent_at else None,
                        'metadata': notification.metadata
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get notification status: {str(e)}")
            return None
    
    async def get_notifications(
        self, 
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get notifications with optional filtering."""
        try:
            notifications = self.notifications
            
            if status:
                notifications = [n for n in notifications if n.status == status]
            
            # Sort by creation time (newest first)
            notifications.sort(key=lambda n: n.created_at, reverse=True)
            
            # Limit results
            notifications = notifications[:limit]
            
            return [
                {
                    'id': n.id,
                    'type': n.type.value,
                    'recipient': n.recipient,
                    'subject': n.subject,
                    'status': n.status,
                    'priority': n.priority.value,
                    'created_at': n.created_at.isoformat(),
                    'sent_at': n.sent_at.isoformat() if n.sent_at else None
                }
                for n in notifications
            ]
            
        except Exception as e:
            logger.error(f"Failed to get notifications: {str(e)}")
            return []
    
    async def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        try:
            total = len(self.notifications)
            sent = len([n for n in self.notifications if n.status == "sent"])
            failed = len([n for n in self.notifications if n.status == "failed"])
            pending = len([n for n in self.notifications if n.status == "pending"])
            
            return {
                'total_notifications': total,
                'sent_notifications': sent,
                'failed_notifications': failed,
                'pending_notifications': pending,
                'success_rate': (sent / total * 100) if total > 0 else 0,
                'retry_queue_size': len(self.retry_queue)
            }
            
        except Exception as e:
            logger.error(f"Failed to get notification stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup notification service."""
        try:
            logger.info("Notification Service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Notification Service: {str(e)}")











