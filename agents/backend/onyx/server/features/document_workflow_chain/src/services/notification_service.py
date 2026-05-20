"""
Notification Service - Fast Implementation
==========================================

Fast notification service with multiple channels.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any, Union
import asyncio
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    """Notification channel enumeration"""
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


class NotificationService:
    """Fast notification service with multiple channels"""
    
    def __init__(self):
        self.channels = {
            NotificationChannel.EMAIL: self._send_email,
            NotificationChannel.SMS: self._send_sms,
            NotificationChannel.PUSH: self._send_push,
            NotificationChannel.WEBHOOK: self._send_webhook,
            NotificationChannel.SLACK: self._send_slack,
            NotificationChannel.TEAMS: self._send_teams,
            NotificationChannel.DISCORD: self._send_discord,
            NotificationChannel.IN_APP: self._send_in_app
        }
        self.templates = {}
        self.notification_history = []
        self.stats = {
            "total_sent": 0,
            "total_delivered": 0,
            "total_failed": 0,
            "by_channel": {channel.value: 0 for channel in NotificationChannel}
        }
    
    async def send_notification(
        self,
        channel: NotificationChannel,
        recipient: str,
        subject: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send notification through specified channel"""
        try:
            # Create notification record
            notification = {
                "id": len(self.notification_history) + 1,
                "channel": channel.value,
                "recipient": recipient,
                "subject": subject,
                "message": message,
                "priority": priority.value,
                "status": NotificationStatus.PENDING.value,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "sent_at": None,
                "delivered_at": None,
                "error": None
            }
            
            # Send notification
            if channel in self.channels:
                result = await self.channels[channel](
                    recipient=recipient,
                    subject=subject,
                    message=message,
                    priority=priority,
                    metadata=metadata
                )
                
                if result.get("success", False):
                    notification["status"] = NotificationStatus.SENT.value
                    notification["sent_at"] = datetime.utcnow().isoformat()
                    self.stats["total_sent"] += 1
                    self.stats["by_channel"][channel.value] += 1
                    
                    # Simulate delivery
                    await asyncio.sleep(0.1)
                    notification["status"] = NotificationStatus.DELIVERED.value
                    notification["delivered_at"] = datetime.utcnow().isoformat()
                    self.stats["total_delivered"] += 1
                else:
                    notification["status"] = NotificationStatus.FAILED.value
                    notification["error"] = result.get("error", "Unknown error")
                    self.stats["total_failed"] += 1
            else:
                notification["status"] = NotificationStatus.FAILED.value
                notification["error"] = f"Unsupported channel: {channel}"
                self.stats["total_failed"] += 1
            
            # Store notification history
            self.notification_history.append(notification)
            
            logger.info(f"Notification sent via {channel.value} to {recipient}")
            return notification
        
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return {
                "success": False,
                "error": str(e),
                "channel": channel.value,
                "recipient": recipient
            }
    
    async def send_template_notification(
        self,
        template_name: str,
        channel: NotificationChannel,
        recipient: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send notification using template"""
        try:
            # Get template
            template = self.templates.get(template_name)
            if not template:
                return {
                    "success": False,
                    "error": f"Template not found: {template_name}"
                }
            
            # Render template
            subject = self._render_template(template.get("subject", ""), variables or {})
            message = self._render_template(template.get("message", ""), variables or {})
            priority = NotificationPriority(template.get("priority", "normal"))
            
            # Send notification
            return await self.send_notification(
                channel=channel,
                recipient=recipient,
                subject=subject,
                message=message,
                priority=priority,
                metadata={"template": template_name, "variables": variables}
            )
        
        except Exception as e:
            logger.error(f"Failed to send template notification: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_bulk_notifications(
        self,
        notifications: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Send multiple notifications in parallel"""
        try:
            tasks = []
            for notification in notifications:
                task = self.send_notification(
                    channel=NotificationChannel(notification["channel"]),
                    recipient=notification["recipient"],
                    subject=notification["subject"],
                    message=notification["message"],
                    priority=NotificationPriority(notification.get("priority", "normal")),
                    metadata=notification.get("metadata")
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "success": False,
                        "error": str(result),
                        "index": i
                    })
                else:
                    processed_results.append(result)
            
            logger.info(f"Bulk notifications sent: {len(processed_results)}")
            return processed_results
        
        except Exception as e:
            logger.error(f"Failed to send bulk notifications: {e}")
            return [{"success": False, "error": str(e)}]
    
    async def _send_email(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: NotificationPriority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send email notification"""
        try:
            # Simulate email sending
            await asyncio.sleep(0.1)
            
            # Mock email service
            return {
                "success": True,
                "message_id": f"email_{datetime.utcnow().timestamp()}",
                "provider": "mock_email_service"
            }
        
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_sms(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: NotificationPriority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send SMS notification"""
        try:
            # Simulate SMS sending
            await asyncio.sleep(0.05)
            
            # Mock SMS service
            return {
                "success": True,
                "message_id": f"sms_{datetime.utcnow().timestamp()}",
                "provider": "mock_sms_service"
            }
        
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_push(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: NotificationPriority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send push notification"""
        try:
            # Simulate push notification
            await asyncio.sleep(0.05)
            
            # Mock push service
            return {
                "success": True,
                "message_id": f"push_{datetime.utcnow().timestamp()}",
                "provider": "mock_push_service"
            }
        
        except Exception as e:
            logger.error(f"Failed to send push notification: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_webhook(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: NotificationPriority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send webhook notification"""
        try:
            # Simulate webhook sending
            await asyncio.sleep(0.1)
            
            # Mock webhook service
            return {
                "success": True,
                "message_id": f"webhook_{datetime.utcnow().timestamp()}",
                "provider": "mock_webhook_service"
            }
        
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_slack(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: NotificationPriority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send Slack notification"""
        try:
            # Simulate Slack sending
            await asyncio.sleep(0.1)
            
            # Mock Slack service
            return {
                "success": True,
                "message_id": f"slack_{datetime.utcnow().timestamp()}",
                "provider": "mock_slack_service"
            }
        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_teams(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: NotificationPriority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send Teams notification"""
        try:
            # Simulate Teams sending
            await asyncio.sleep(0.1)
            
            # Mock Teams service
            return {
                "success": True,
                "message_id": f"teams_{datetime.utcnow().timestamp()}",
                "provider": "mock_teams_service"
            }
        
        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_discord(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: NotificationPriority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send Discord notification"""
        try:
            # Simulate Discord sending
            await asyncio.sleep(0.1)
            
            # Mock Discord service
            return {
                "success": True,
                "message_id": f"discord_{datetime.utcnow().timestamp()}",
                "provider": "mock_discord_service"
            }
        
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_in_app(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: NotificationPriority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send in-app notification"""
        try:
            # Simulate in-app notification
            await asyncio.sleep(0.01)
            
            # Mock in-app service
            return {
                "success": True,
                "message_id": f"in_app_{datetime.utcnow().timestamp()}",
                "provider": "mock_in_app_service"
            }
        
        except Exception as e:
            logger.error(f"Failed to send in-app notification: {e}")
            return {"success": False, "error": str(e)}
    
    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Render template with variables"""
        try:
            rendered = template
            for key, value in variables.items():
                placeholder = f"{{{key}}}"
                rendered = rendered.replace(placeholder, str(value))
            return rendered
        
        except Exception as e:
            logger.error(f"Failed to render template: {e}")
            return template
    
    def add_template(self, name: str, template: Dict[str, Any]) -> bool:
        """Add notification template"""
        try:
            self.templates[name] = template
            logger.info(f"Template added: {name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add template: {e}")
            return False
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get notification template"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List available templates"""
        return list(self.templates.keys())
    
    async def get_notification_history(
        self,
        limit: int = 100,
        channel: Optional[NotificationChannel] = None,
        status: Optional[NotificationStatus] = None
    ) -> List[Dict[str, Any]]:
        """Get notification history"""
        try:
            history = self.notification_history.copy()
            
            # Filter by channel
            if channel:
                history = [n for n in history if n["channel"] == channel.value]
            
            # Filter by status
            if status:
                history = [n for n in history if n["status"] == status.value]
            
            # Sort by created_at (newest first)
            history.sort(key=lambda x: x["created_at"], reverse=True)
            
            # Limit results
            return history[:limit]
        
        except Exception as e:
            logger.error(f"Failed to get notification history: {e}")
            return []
    
    async def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        try:
            total_notifications = len(self.notification_history)
            success_rate = (self.stats["total_delivered"] / total_notifications * 100) if total_notifications > 0 else 0
            
            return {
                "total_notifications": total_notifications,
                "total_sent": self.stats["total_sent"],
                "total_delivered": self.stats["total_delivered"],
                "total_failed": self.stats["total_failed"],
                "success_rate": round(success_rate, 2),
                "by_channel": self.stats["by_channel"],
                "available_templates": len(self.templates),
                "available_channels": [channel.value for channel in NotificationChannel],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get notification stats: {e}")
            return {"error": str(e)}


# Global notification service instance
notification_service = NotificationService()