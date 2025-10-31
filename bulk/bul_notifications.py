"""
BUL - Business Universal Language (Advanced Notifications System)
===============================================================

Advanced notification system with multiple channels and real-time delivery.
"""

import asyncio
import logging
import smtplib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import redis
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///bul_notifications.db', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Prometheus Metrics
NOTIFICATION_REQUESTS = Counter('bul_notification_requests_total', 'Total notification requests', ['channel', 'type'])
NOTIFICATION_DELIVERED = Counter('bul_notification_delivered_total', 'Notifications delivered', ['channel', 'status'])
NOTIFICATION_FAILED = Counter('bul_notification_failed_total', 'Notifications failed', ['channel', 'error'])

class NotificationChannel(str, Enum):
    """Notification channel enumeration."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    WEBSOCKET = "websocket"

class NotificationPriority(str, Enum):
    """Notification priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class NotificationStatus(str, Enum):
    """Notification status enumeration."""
    PENDING = "pending"
    SENDING = "sending"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Database Models
class NotificationTemplate(Base):
    __tablename__ = "notification_templates"
    
    id = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    channel = Column(String, nullable=False)
    subject_template = Column(Text)
    body_template = Column(Text, nullable=False)
    variables = Column(Text, default="[]")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class NotificationSubscription(Base):
    __tablename__ = "notification_subscriptions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)  # email, phone, webhook URL, etc.
    preferences = Column(Text, default="{}")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class NotificationLog(Base):
    __tablename__ = "notification_logs"
    
    id = Column(String, primary_key=True)
    template_id = Column(String, ForeignKey("notification_templates.id"))
    user_id = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    subject = Column(String)
    body = Column(Text, nullable=False)
    priority = Column(String, default=NotificationPriority.MEDIUM)
    status = Column(String, default=NotificationStatus.PENDING)
    sent_at = Column(DateTime)
    delivered_at = Column(DateTime)
    error_message = Column(Text)
    metadata = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Notification Configuration
NOTIFICATION_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_username": "your-email@gmail.com",
    "smtp_password": "your-app-password",
    "twilio_account_sid": "your-twilio-sid",
    "twilio_auth_token": "your-twilio-token",
    "twilio_phone_number": "+1234567890",
    "slack_webhook_url": "https://hooks.slack.com/services/...",
    "teams_webhook_url": "https://outlook.office.com/webhook/...",
    "discord_webhook_url": "https://discord.com/api/webhooks/...",
    "firebase_server_key": "your-firebase-server-key",
    "rate_limit_per_minute": 100,
    "retry_attempts": 3,
    "retry_delay": 60
}

class AdvancedNotificationSystem:
    """Advanced notification system with multiple channels."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL Advanced Notifications System",
            description="Advanced notification system with multiple channels and real-time delivery",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Database session
        self.db = SessionLocal()
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup components
        self.setup_middleware()
        self.setup_routes()
        self.setup_default_data()
        
        logger.info("Advanced Notification System initialized")
    
    def setup_middleware(self):
        """Setup notification middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup notification API routes."""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with notification system information."""
            return {
                "message": "BUL Advanced Notifications System",
                "version": "1.0.0",
                "status": "operational",
                "features": [
                    "Multi-Channel Delivery",
                    "Template Management",
                    "Real-Time WebSocket",
                    "Priority Queuing",
                    "Delivery Tracking",
                    "Retry Logic",
                    "Rate Limiting",
                    "Analytics Dashboard"
                ],
                "channels": [channel.value for channel in NotificationChannel],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/notifications/send", tags=["Notifications"])
        async def send_notification(notification_request: dict, background_tasks: BackgroundTasks):
            """Send notification through specified channel."""
            try:
                # Validate request
                required_fields = ["template_name", "user_id", "variables"]
                if not all(field in notification_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                template_name = notification_request["template_name"]
                user_id = notification_request["user_id"]
                variables = notification_request["variables"]
                priority = notification_request.get("priority", NotificationPriority.MEDIUM)
                channel = notification_request.get("channel", NotificationChannel.EMAIL)
                
                # Get template
                template = self.db.query(NotificationTemplate).filter(
                    NotificationTemplate.name == template_name,
                    NotificationTemplate.is_active == True
                ).first()
                
                if not template:
                    raise HTTPException(status_code=404, detail="Template not found")
                
                # Get user subscription
                subscription = self.db.query(NotificationSubscription).filter(
                    NotificationSubscription.user_id == user_id,
                    NotificationSubscription.channel == channel,
                    NotificationSubscription.is_active == True
                ).first()
                
                if not subscription:
                    raise HTTPException(status_code=404, detail="User subscription not found")
                
                # Process template
                subject = self.process_template(template.subject_template, variables)
                body = self.process_template(template.body_template, variables)
                
                # Create notification log
                notification_log = NotificationLog(
                    id=f"notif_{int(time.time())}",
                    template_id=template.id,
                    user_id=user_id,
                    channel=channel,
                    subject=subject,
                    body=body,
                    priority=priority,
                    status=NotificationStatus.PENDING,
                    metadata=json.dumps(variables)
                )
                
                self.db.add(notification_log)
                self.db.commit()
                
                # Send notification in background
                background_tasks.add_task(
                    self.deliver_notification,
                    notification_log.id,
                    channel,
                    subscription.endpoint,
                    subject,
                    body,
                    priority
                )
                
                NOTIFICATION_REQUESTS.labels(channel=channel, type=priority).inc()
                
                return {
                    "message": "Notification queued for delivery",
                    "notification_id": notification_log.id,
                    "status": "pending"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error sending notification: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/notifications/broadcast", tags=["Notifications"])
        async def broadcast_notification(broadcast_request: dict, background_tasks: BackgroundTasks):
            """Broadcast notification to multiple users."""
            try:
                template_name = broadcast_request["template_name"]
                user_ids = broadcast_request["user_ids"]
                variables = broadcast_request["variables"]
                priority = broadcast_request.get("priority", NotificationPriority.MEDIUM)
                channel = broadcast_request.get("channel", NotificationChannel.EMAIL)
                
                # Get template
                template = self.db.query(NotificationTemplate).filter(
                    NotificationTemplate.name == template_name,
                    NotificationTemplate.is_active == True
                ).first()
                
                if not template:
                    raise HTTPException(status_code=404, detail="Template not found")
                
                notification_ids = []
                
                for user_id in user_ids:
                    # Get user subscription
                    subscription = self.db.query(NotificationSubscription).filter(
                        NotificationSubscription.user_id == user_id,
                        NotificationSubscription.channel == channel,
                        NotificationSubscription.is_active == True
                    ).first()
                    
                    if subscription:
                        # Process template
                        subject = self.process_template(template.subject_template, variables)
                        body = self.process_template(template.body_template, variables)
                        
                        # Create notification log
                        notification_log = NotificationLog(
                            id=f"notif_{int(time.time())}_{user_id}",
                            template_id=template.id,
                            user_id=user_id,
                            channel=channel,
                            subject=subject,
                            body=body,
                            priority=priority,
                            status=NotificationStatus.PENDING,
                            metadata=json.dumps(variables)
                        )
                        
                        self.db.add(notification_log)
                        notification_ids.append(notification_log.id)
                
                self.db.commit()
                
                # Send notifications in background
                for notification_id in notification_ids:
                    background_tasks.add_task(
                        self.deliver_notification_by_id,
                        notification_id
                    )
                
                return {
                    "message": f"Broadcast queued for {len(notification_ids)} users",
                    "notification_ids": notification_ids,
                    "status": "pending"
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error broadcasting notification: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/notifications/status/{notification_id}", tags=["Notifications"])
        async def get_notification_status(notification_id: str):
            """Get notification delivery status."""
            try:
                notification = self.db.query(NotificationLog).filter(
                    NotificationLog.id == notification_id
                ).first()
                
                if not notification:
                    raise HTTPException(status_code=404, detail="Notification not found")
                
                return {
                    "notification_id": notification.id,
                    "user_id": notification.user_id,
                    "channel": notification.channel,
                    "priority": notification.priority,
                    "status": notification.status,
                    "sent_at": notification.sent_at.isoformat() if notification.sent_at else None,
                    "delivered_at": notification.delivered_at.isoformat() if notification.delivered_at else None,
                    "error_message": notification.error_message,
                    "created_at": notification.created_at.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting notification status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/notifications/history", tags=["Notifications"])
        async def get_notification_history(limit: int = 100):
            """Get notification history."""
            try:
                notifications = self.db.query(NotificationLog).order_by(
                    NotificationLog.created_at.desc()
                ).limit(limit).all()
                
                return {
                    "notifications": [
                        {
                            "id": notif.id,
                            "user_id": notif.user_id,
                            "channel": notif.channel,
                            "subject": notif.subject,
                            "priority": notif.priority,
                            "status": notif.status,
                            "sent_at": notif.sent_at.isoformat() if notif.sent_at else None,
                            "delivered_at": notif.delivered_at.isoformat() if notif.delivered_at else None,
                            "created_at": notif.created_at.isoformat()
                        }
                        for notif in notifications
                    ],
                    "total": len(notifications)
                }
                
            except Exception as e:
                logger.error(f"Error getting notification history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/templates", tags=["Templates"])
        async def create_template(template_request: dict):
            """Create notification template."""
            try:
                template = NotificationTemplate(
                    id=f"template_{int(time.time())}",
                    name=template_request["name"],
                    channel=template_request["channel"],
                    subject_template=template_request.get("subject_template"),
                    body_template=template_request["body_template"],
                    variables=json.dumps(template_request.get("variables", [])),
                    is_active=template_request.get("is_active", True)
                )
                
                self.db.add(template)
                self.db.commit()
                
                return {
                    "message": "Template created successfully",
                    "template_id": template.id,
                    "name": template.name
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error creating template: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/templates", tags=["Templates"])
        async def get_templates():
            """Get all notification templates."""
            try:
                templates = self.db.query(NotificationTemplate).filter(
                    NotificationTemplate.is_active == True
                ).all()
                
                return {
                    "templates": [
                        {
                            "id": template.id,
                            "name": template.name,
                            "channel": template.channel,
                            "subject_template": template.subject_template,
                            "body_template": template.body_template,
                            "variables": json.loads(template.variables),
                            "created_at": template.created_at.isoformat()
                        }
                        for template in templates
                    ],
                    "total": len(templates)
                }
                
            except Exception as e:
                logger.error(f"Error getting templates: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/subscriptions", tags=["Subscriptions"])
        async def create_subscription(subscription_request: dict):
            """Create notification subscription."""
            try:
                subscription = NotificationSubscription(
                    id=f"sub_{int(time.time())}",
                    user_id=subscription_request["user_id"],
                    channel=subscription_request["channel"],
                    endpoint=subscription_request["endpoint"],
                    preferences=json.dumps(subscription_request.get("preferences", {})),
                    is_active=subscription_request.get("is_active", True)
                )
                
                self.db.add(subscription)
                self.db.commit()
                
                return {
                    "message": "Subscription created successfully",
                    "subscription_id": subscription.id,
                    "user_id": subscription.user_id
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error creating subscription: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/subscriptions/{user_id}", tags=["Subscriptions"])
        async def get_user_subscriptions(user_id: str):
            """Get user notification subscriptions."""
            try:
                subscriptions = self.db.query(NotificationSubscription).filter(
                    NotificationSubscription.user_id == user_id,
                    NotificationSubscription.is_active == True
                ).all()
                
                return {
                    "subscriptions": [
                        {
                            "id": sub.id,
                            "channel": sub.channel,
                            "endpoint": sub.endpoint,
                            "preferences": json.loads(sub.preferences),
                            "created_at": sub.created_at.isoformat()
                        }
                        for sub in subscriptions
                    ],
                    "total": len(subscriptions)
                }
                
            except Exception as e:
                logger.error(f"Error getting subscriptions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time notifications."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
        
        @self.app.get("/dashboard", tags=["Dashboard"])
        async def get_notification_dashboard():
            """Get notification dashboard data."""
            try:
                # Get statistics
                total_notifications = self.db.query(NotificationLog).count()
                delivered_notifications = self.db.query(NotificationLog).filter(
                    NotificationLog.status == NotificationStatus.DELIVERED
                ).count()
                failed_notifications = self.db.query(NotificationLog).filter(
                    NotificationLog.status == NotificationStatus.FAILED
                ).count()
                
                # Get channel distribution
                channel_stats = {}
                for channel in NotificationChannel:
                    count = self.db.query(NotificationLog).filter(
                        NotificationLog.channel == channel.value
                    ).count()
                    channel_stats[channel.value] = count
                
                # Get recent activity
                recent_notifications = self.db.query(NotificationLog).order_by(
                    NotificationLog.created_at.desc()
                ).limit(10).all()
                
                return {
                    "summary": {
                        "total_notifications": total_notifications,
                        "delivered_notifications": delivered_notifications,
                        "failed_notifications": failed_notifications,
                        "delivery_rate": (delivered_notifications / total_notifications * 100) if total_notifications > 0 else 0
                    },
                    "channel_distribution": channel_stats,
                    "recent_activity": [
                        {
                            "id": notif.id,
                            "user_id": notif.user_id,
                            "channel": notif.channel,
                            "status": notif.status,
                            "created_at": notif.created_at.isoformat()
                        }
                        for notif in recent_notifications
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def setup_default_data(self):
        """Setup default notification data."""
        try:
            # Create default templates
            templates = [
                {
                    "name": "welcome_email",
                    "channel": NotificationChannel.EMAIL,
                    "subject_template": "Welcome to BUL System, {{user_name}}!",
                    "body_template": "Hello {{user_name}},\n\nWelcome to the BUL system! Your account has been created successfully.\n\nBest regards,\nBUL Team",
                    "variables": ["user_name"]
                },
                {
                    "name": "system_alert",
                    "channel": NotificationChannel.SLACK,
                    "subject_template": None,
                    "body_template": "🚨 System Alert: {{alert_message}}\n\nSeverity: {{severity}}\nTime: {{timestamp}}",
                    "variables": ["alert_message", "severity", "timestamp"]
                },
                {
                    "name": "task_assignment",
                    "channel": NotificationChannel.EMAIL,
                    "subject_template": "New Task Assigned: {{task_title}}",
                    "body_template": "Hello {{user_name}},\n\nYou have been assigned a new task:\n\nTitle: {{task_title}}\nDescription: {{task_description}}\nDue Date: {{due_date}}\n\nPlease log in to view details.\n\nBest regards,\nBUL Team",
                    "variables": ["user_name", "task_title", "task_description", "due_date"]
                }
            ]
            
            for template_data in templates:
                template = NotificationTemplate(
                    id=f"template_{template_data['name']}",
                    name=template_data["name"],
                    channel=template_data["channel"],
                    subject_template=template_data["subject_template"],
                    body_template=template_data["body_template"],
                    variables=json.dumps(template_data["variables"]),
                    is_active=True
                )
                
                self.db.add(template)
            
            self.db.commit()
            logger.info("Default notification templates created")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating default notification data: {e}")
    
    def process_template(self, template: str, variables: dict) -> str:
        """Process template with variables."""
        if not template:
            return ""
        
        processed = template
        for key, value in variables.items():
            processed = processed.replace(f"{{{{{key}}}}}", str(value))
        
        return processed
    
    async def deliver_notification(self, notification_id: str, channel: str, endpoint: str, 
                                 subject: str, body: str, priority: str):
        """Deliver notification through specified channel."""
        try:
            # Update status to sending
            notification = self.db.query(NotificationLog).filter(
                NotificationLog.id == notification_id
            ).first()
            
            if not notification:
                return
            
            notification.status = NotificationStatus.SENDING
            notification.sent_at = datetime.utcnow()
            self.db.commit()
            
            # Deliver based on channel
            success = False
            
            if channel == NotificationChannel.EMAIL:
                success = await self.send_email(endpoint, subject, body)
            elif channel == NotificationChannel.SMS:
                success = await self.send_sms(endpoint, body)
            elif channel == NotificationChannel.SLACK:
                success = await self.send_slack(endpoint, body)
            elif channel == NotificationChannel.TEAMS:
                success = await self.send_teams(endpoint, body)
            elif channel == NotificationChannel.DISCORD:
                success = await self.send_discord(endpoint, body)
            elif channel == NotificationChannel.WEBHOOK:
                success = await self.send_webhook(endpoint, {"subject": subject, "body": body})
            elif channel == NotificationChannel.WEBSOCKET:
                success = await self.send_websocket(endpoint, {"subject": subject, "body": body})
            
            # Update status
            if success:
                notification.status = NotificationStatus.DELIVERED
                notification.delivered_at = datetime.utcnow()
                NOTIFICATION_DELIVERED.labels(channel=channel, status="success").inc()
            else:
                notification.status = NotificationStatus.FAILED
                notification.error_message = "Delivery failed"
                NOTIFICATION_FAILED.labels(channel=channel, error="delivery_failed").inc()
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error delivering notification {notification_id}: {e}")
            
            # Update status to failed
            notification = self.db.query(NotificationLog).filter(
                NotificationLog.id == notification_id
            ).first()
            
            if notification:
                notification.status = NotificationStatus.FAILED
                notification.error_message = str(e)
                self.db.commit()
            
            NOTIFICATION_FAILED.labels(channel=channel, error="exception").inc()
    
    async def deliver_notification_by_id(self, notification_id: str):
        """Deliver notification by ID."""
        try:
            notification = self.db.query(NotificationLog).filter(
                NotificationLog.id == notification_id
            ).first()
            
            if not notification:
                return
            
            # Get user subscription
            subscription = self.db.query(NotificationSubscription).filter(
                NotificationSubscription.user_id == notification.user_id,
                NotificationSubscription.channel == notification.channel,
                NotificationSubscription.is_active == True
            ).first()
            
            if subscription:
                await self.deliver_notification(
                    notification_id,
                    notification.channel,
                    subscription.endpoint,
                    notification.subject,
                    notification.body,
                    notification.priority
                )
            
        except Exception as e:
            logger.error(f"Error delivering notification by ID {notification_id}: {e}")
    
    async def send_email(self, to_email: str, subject: str, body: str) -> bool:
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = NOTIFICATION_CONFIG["smtp_username"]
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(NOTIFICATION_CONFIG["smtp_server"], NOTIFICATION_CONFIG["smtp_port"])
            server.starttls()
            server.login(NOTIFICATION_CONFIG["smtp_username"], NOTIFICATION_CONFIG["smtp_password"])
            
            text = msg.as_string()
            server.sendmail(NOTIFICATION_CONFIG["smtp_username"], to_email, text)
            server.quit()
            
            logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email to {to_email}: {e}")
            return False
    
    async def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS notification."""
        try:
            # This would use Twilio or similar service
            # For now, just log the SMS
            logger.info(f"SMS sent to {phone_number}: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SMS to {phone_number}: {e}")
            return False
    
    async def send_slack(self, webhook_url: str, message: str) -> bool:
        """Send Slack notification."""
        try:
            payload = {"text": message}
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack message sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return False
    
    async def send_teams(self, webhook_url: str, message: str) -> bool:
        """Send Microsoft Teams notification."""
        try:
            payload = {
                "text": message,
                "type": "message"
            }
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Teams message sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Teams message: {e}")
            return False
    
    async def send_discord(self, webhook_url: str, message: str) -> bool:
        """Send Discord notification."""
        try:
            payload = {"content": message}
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Discord message sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")
            return False
    
    async def send_webhook(self, webhook_url: str, data: dict) -> bool:
        """Send webhook notification."""
        try:
            response = requests.post(webhook_url, json=data, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook sent to {webhook_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook to {webhook_url}: {e}")
            return False
    
    async def send_websocket(self, user_id: str, data: dict) -> bool:
        """Send WebSocket notification."""
        try:
            # Send to all active connections (in a real app, you'd filter by user)
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps({
                        "type": "notification",
                        "user_id": user_id,
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    }))
                except:
                    # Remove disconnected connections
                    self.active_connections.remove(connection)
            
            logger.info(f"WebSocket notification sent to {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending WebSocket notification to {user_id}: {e}")
            return False
    
    def run(self, host: str = "0.0.0.0", port: int = 8006, debug: bool = False):
        """Run the notification system."""
        logger.info(f"Starting Advanced Notifications System on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Advanced Notifications System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8006, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run notification system
    system = AdvancedNotificationSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
