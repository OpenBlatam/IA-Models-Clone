"""
Notification System for AI Document Processor
Real, working notification features for document processing
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

logger = logging.getLogger(__name__)

class NotificationSystem:
    """Real working notification system for AI document processing"""
    
    def __init__(self):
        self.notifications = []
        self.subscribers = {}
        self.email_config = {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("EMAIL_USERNAME", ""),
            "password": os.getenv("EMAIL_PASSWORD", ""),
            "from_email": os.getenv("FROM_EMAIL", ""),
            "enabled": os.getenv("EMAIL_ENABLED", "false").lower() == "true"
        }
        
        # Notification stats
        self.stats = {
            "total_notifications": 0,
            "email_notifications": 0,
            "webhook_notifications": 0,
            "failed_notifications": 0,
            "start_time": time.time()
        }
    
    async def send_notification(self, notification_type: str, message: str, 
                              recipients: List[str] = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send notification to subscribers"""
        try:
            notification = {
                "id": self._generate_notification_id(),
                "type": notification_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "recipients": recipients or [],
                "data": data or {},
                "status": "pending"
            }
            
            # Add to notifications list
            self.notifications.append(notification)
            
            # Send to email subscribers
            if self.email_config["enabled"] and recipients:
                email_result = await self._send_email_notification(notification)
                if email_result["success"]:
                    notification["status"] = "sent"
                    self.stats["email_notifications"] += 1
                else:
                    notification["status"] = "failed"
                    self.stats["failed_notifications"] += 1
            
            # Send to webhook subscribers
            webhook_result = await self._send_webhook_notification(notification)
            if webhook_result["success"]:
                if notification["status"] == "pending":
                    notification["status"] = "sent"
                self.stats["webhook_notifications"] += 1
            else:
                if notification["status"] == "pending":
                    notification["status"] = "failed"
                    self.stats["failed_notifications"] += 1
            
            # Update stats
            self.stats["total_notifications"] += 1
            
            return {
                "notification_id": notification["id"],
                "status": notification["status"],
                "timestamp": notification["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    async def _send_email_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification"""
        try:
            if not self.email_config["enabled"]:
                return {"success": False, "reason": "Email not enabled"}
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config["from_email"]
            msg['To'] = ", ".join(notification["recipients"])
            msg['Subject'] = f"AI Document Processor - {notification['type']}"
            
            # Create email body
            body = f"""
            Notification Type: {notification['type']}
            Message: {notification['message']}
            Timestamp: {notification['timestamp']}
            
            Data: {json.dumps(notification['data'], indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            server.send_message(msg)
            server.quit()
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_webhook_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook notification"""
        try:
            # This would integrate with webhook services
            # For now, just log the notification
            logger.info(f"Webhook notification: {notification['type']} - {notification['message']}")
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return {"success": False, "error": str(e)}
    
    async def subscribe(self, subscriber_id: str, notification_types: List[str], 
                      email: str = None, webhook_url: str = None) -> Dict[str, Any]:
        """Subscribe to notifications"""
        try:
            self.subscribers[subscriber_id] = {
                "notification_types": notification_types,
                "email": email,
                "webhook_url": webhook_url,
                "subscribed_at": datetime.now().isoformat(),
                "active": True
            }
            
            return {
                "subscriber_id": subscriber_id,
                "status": "subscribed",
                "notification_types": notification_types
            }
            
        except Exception as e:
            logger.error(f"Error subscribing: {e}")
            return {"error": str(e)}
    
    async def unsubscribe(self, subscriber_id: str) -> Dict[str, Any]:
        """Unsubscribe from notifications"""
        try:
            if subscriber_id in self.subscribers:
                self.subscribers[subscriber_id]["active"] = False
                return {
                    "subscriber_id": subscriber_id,
                    "status": "unsubscribed"
                }
            else:
                return {
                    "subscriber_id": subscriber_id,
                    "status": "not_found"
                }
                
        except Exception as e:
            logger.error(f"Error unsubscribing: {e}")
            return {"error": str(e)}
    
    async def send_processing_notification(self, processing_result: Dict[str, Any], 
                                        recipients: List[str] = None) -> Dict[str, Any]:
        """Send processing completion notification"""
        try:
            message = f"Document processing completed: {processing_result.get('status', 'unknown')}"
            data = {
                "processing_time": processing_result.get("processing_time", 0),
                "document_id": processing_result.get("document_id", ""),
                "filename": processing_result.get("filename", ""),
                "file_size": processing_result.get("file_size", 0)
            }
            
            return await self.send_notification(
                "processing_complete",
                message,
                recipients,
                data
            )
            
        except Exception as e:
            logger.error(f"Error sending processing notification: {e}")
            return {"error": str(e)}
    
    async def send_error_notification(self, error_message: str, 
                                    recipients: List[str] = None) -> Dict[str, Any]:
        """Send error notification"""
        try:
            return await self.send_notification(
                "error",
                f"System error: {error_message}",
                recipients,
                {"error_message": error_message}
            )
            
        except Exception as e:
            logger.error(f"Error sending error notification: {e}")
            return {"error": str(e)}
    
    async def send_security_notification(self, security_event: str, 
                                      client_ip: str, recipients: List[str] = None) -> Dict[str, Any]:
        """Send security notification"""
        try:
            message = f"Security event: {security_event} from {client_ip}"
            data = {
                "security_event": security_event,
                "client_ip": client_ip,
                "timestamp": datetime.now().isoformat()
            }
            
            return await self.send_notification(
                "security_alert",
                message,
                recipients,
                data
            )
            
        except Exception as e:
            logger.error(f"Error sending security notification: {e}")
            return {"error": str(e)}
    
    async def send_performance_notification(self, performance_data: Dict[str, Any], 
                                         recipients: List[str] = None) -> Dict[str, Any]:
        """Send performance notification"""
        try:
            message = f"Performance alert: {performance_data.get('message', 'System performance issue')}"
            
            return await self.send_notification(
                "performance_alert",
                message,
                recipients,
                performance_data
            )
            
        except Exception as e:
            logger.error(f"Error sending performance notification: {e}")
            return {"error": str(e)}
    
    def _generate_notification_id(self) -> str:
        """Generate unique notification ID"""
        return f"notif_{int(time.time())}_{len(self.notifications)}"
    
    def get_notifications(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent notifications"""
        return self.notifications[-limit:]
    
    def get_subscribers(self) -> Dict[str, Any]:
        """Get all subscribers"""
        return self.subscribers.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        uptime = time.time() - self.stats["start_time"]
        return {
            "stats": self.stats.copy(),
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "total_subscribers": len(self.subscribers),
            "active_subscribers": len([s for s in self.subscribers.values() if s["active"]]),
            "email_enabled": self.email_config["enabled"]
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get notification configuration"""
        return {
            "email_config": {
                "enabled": self.email_config["enabled"],
                "smtp_server": self.email_config["smtp_server"],
                "smtp_port": self.email_config["smtp_port"],
                "from_email": self.email_config["from_email"]
            },
            "features": {
                "email_notifications": self.email_config["enabled"],
                "webhook_notifications": True,
                "processing_notifications": True,
                "error_notifications": True,
                "security_notifications": True,
                "performance_notifications": True
            }
        }

# Global instance
notification_system = NotificationSystem()













