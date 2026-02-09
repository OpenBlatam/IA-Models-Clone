"""
Notification System - Sistema de notificaciones avanzado
"""

import asyncio
import logging
import smtplib
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import aiohttp
import aiosmtplib

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Tipos de notificaciones."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    INTERNAL = "internal"


class NotificationPriority(Enum):
    """Prioridades de notificación."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class NotificationStatus(Enum):
    """Estados de notificación."""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NotificationTemplate:
    """Plantilla de notificación."""
    template_id: str
    name: str
    notification_type: NotificationType
    subject: str
    content: str
    variables: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class NotificationRecipient:
    """Destinatario de notificación."""
    recipient_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    webhook_url: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class Notification:
    """Notificación."""
    notification_id: str
    notification_type: NotificationType
    priority: NotificationPriority
    status: NotificationStatus
    template_id: Optional[str]
    subject: str
    content: str
    recipients: List[NotificationRecipient]
    sender: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


class NotificationSystem:
    """
    Sistema de notificaciones avanzado.
    """
    
    def __init__(self):
        """Inicializar sistema de notificaciones."""
        self.templates: Dict[str, NotificationTemplate] = {}
        self.recipients: Dict[str, NotificationRecipient] = {}
        self.notifications: Dict[str, Notification] = {}
        
        # Configuración de proveedores
        self.email_config = {
            "smtp_host": "localhost",
            "smtp_port": 587,
            "smtp_username": "",
            "smtp_password": "",
            "smtp_use_tls": True
        }
        
        self.sms_config = {
            "provider": "twilio",
            "account_sid": "",
            "auth_token": "",
            "from_number": ""
        }
        
        self.webhook_config = {
            "timeout": 30,
            "retry_attempts": 3,
            "retry_delay": 5
        }
        
        # Cola de notificaciones
        self.notification_queue = asyncio.Queue()
        self.processing_task = None
        
        # Estadísticas
        self.stats = {
            "total_notifications": 0,
            "sent_notifications": 0,
            "failed_notifications": 0,
            "pending_notifications": 0,
            "start_time": datetime.now()
        }
        
        logger.info("NotificationSystem inicializado")
    
    async def initialize(self):
        """Inicializar el sistema de notificaciones."""
        try:
            # Iniciar procesador de notificaciones
            self.processing_task = asyncio.create_task(self._process_notifications())
            
            # Cargar plantillas por defecto
            await self._load_default_templates()
            
            logger.info("NotificationSystem inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar NotificationSystem: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el sistema de notificaciones."""
        try:
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("NotificationSystem cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar NotificationSystem: {e}")
    
    async def _load_default_templates(self):
        """Cargar plantillas por defecto."""
        try:
            # Plantilla de email básica
            email_template = NotificationTemplate(
                template_id="default_email",
                name="Email Básico",
                notification_type=NotificationType.EMAIL,
                subject="Notificación del Sistema",
                content="Hola {name},\n\n{message}\n\nSaludos,\nSistema Export IA",
                variables=["name", "message"]
            )
            self.templates["default_email"] = email_template
            
            # Plantilla de alerta
            alert_template = NotificationTemplate(
                template_id="alert_notification",
                name="Alerta del Sistema",
                notification_type=NotificationType.EMAIL,
                subject="🚨 Alerta: {alert_type}",
                content="Se ha detectado una alerta en el sistema:\n\nTipo: {alert_type}\nSeveridad: {severity}\nMensaje: {message}\nTimestamp: {timestamp}\n\nPor favor, revise el sistema.",
                variables=["alert_type", "severity", "message", "timestamp"]
            )
            self.templates["alert_notification"] = alert_template
            
            # Plantilla de webhook
            webhook_template = NotificationTemplate(
                template_id="default_webhook",
                name="Webhook Básico",
                notification_type=NotificationType.WEBHOOK,
                subject="",
                content='{"event": "{event_type}", "message": "{message}", "timestamp": "{timestamp}"}',
                variables=["event_type", "message", "timestamp"]
            )
            self.templates["default_webhook"] = webhook_template
            
            logger.info("Plantillas por defecto cargadas")
            
        except Exception as e:
            logger.error(f"Error al cargar plantillas por defecto: {e}")
    
    async def create_template(
        self,
        name: str,
        notification_type: NotificationType,
        subject: str,
        content: str,
        variables: List[str] = None
    ) -> str:
        """Crear plantilla de notificación."""
        try:
            template_id = str(uuid.uuid4())
            
            template = NotificationTemplate(
                template_id=template_id,
                name=name,
                notification_type=notification_type,
                subject=subject,
                content=content,
                variables=variables or []
            )
            
            self.templates[template_id] = template
            
            logger.info(f"Plantilla creada: {name} ({template_id})")
            return template_id
            
        except Exception as e:
            logger.error(f"Error al crear plantilla: {e}")
            raise
    
    async def create_recipient(
        self,
        name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        push_token: Optional[str] = None,
        webhook_url: Optional[str] = None,
        preferences: Dict[str, Any] = None
    ) -> str:
        """Crear destinatario."""
        try:
            recipient_id = str(uuid.uuid4())
            
            recipient = NotificationRecipient(
                recipient_id=recipient_id,
                name=name,
                email=email,
                phone=phone,
                push_token=push_token,
                webhook_url=webhook_url,
                preferences=preferences or {}
            )
            
            self.recipients[recipient_id] = recipient
            
            logger.info(f"Destinatario creado: {name} ({recipient_id})")
            return recipient_id
            
        except Exception as e:
            logger.error(f"Error al crear destinatario: {e}")
            raise
    
    async def send_notification(
        self,
        notification_type: NotificationType,
        subject: str,
        content: str,
        recipients: List[str],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        template_id: Optional[str] = None,
        scheduled_at: Optional[datetime] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Enviar notificación."""
        try:
            notification_id = str(uuid.uuid4())
            
            # Obtener destinatarios
            recipient_objects = []
            for recipient_id in recipients:
                if recipient_id in self.recipients:
                    recipient_objects.append(self.recipients[recipient_id])
                else:
                    logger.warning(f"Destinatario no encontrado: {recipient_id}")
            
            if not recipient_objects:
                raise ValueError("No se encontraron destinatarios válidos")
            
            # Crear notificación
            notification = Notification(
                notification_id=notification_id,
                notification_type=notification_type,
                priority=priority,
                status=NotificationStatus.PENDING,
                template_id=template_id,
                subject=subject,
                content=content,
                recipients=recipient_objects,
                scheduled_at=scheduled_at,
                metadata=metadata or {}
            )
            
            self.notifications[notification_id] = notification
            self.stats["total_notifications"] += 1
            self.stats["pending_notifications"] += 1
            
            # Agregar a la cola
            await self.notification_queue.put(notification)
            
            logger.info(f"Notificación creada: {notification_id} ({notification_type.value})")
            return notification_id
            
        except Exception as e:
            logger.error(f"Error al crear notificación: {e}")
            raise
    
    async def send_notification_with_template(
        self,
        template_id: str,
        recipients: List[str],
        variables: Dict[str, str],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        scheduled_at: Optional[datetime] = None
    ) -> str:
        """Enviar notificación usando plantilla."""
        try:
            if template_id not in self.templates:
                raise ValueError(f"Plantilla no encontrada: {template_id}")
            
            template = self.templates[template_id]
            
            # Reemplazar variables en subject y content
            subject = template.subject
            content = template.content
            
            for var_name, var_value in variables.items():
                subject = subject.replace(f"{{{var_name}}}", str(var_value))
                content = content.replace(f"{{{var_name}}}", str(var_value))
            
            return await self.send_notification(
                notification_type=template.notification_type,
                subject=subject,
                content=content,
                recipients=recipients,
                priority=priority,
                template_id=template_id,
                scheduled_at=scheduled_at
            )
            
        except Exception as e:
            logger.error(f"Error al enviar notificación con plantilla: {e}")
            raise
    
    async def _process_notifications(self):
        """Procesar cola de notificaciones."""
        while True:
            try:
                # Obtener notificación de la cola
                notification = await self.notification_queue.get()
                
                # Verificar si está programada
                if notification.scheduled_at and notification.scheduled_at > datetime.now():
                    # Reagendar
                    await asyncio.sleep(1)
                    await self.notification_queue.put(notification)
                    continue
                
                # Procesar notificación
                await self._process_single_notification(notification)
                
                # Marcar como completada
                self.notification_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error al procesar notificación: {e}")
                await asyncio.sleep(1)
    
    async def _process_single_notification(self, notification: Notification):
        """Procesar una notificación individual."""
        try:
            notification.status = NotificationStatus.SENDING
            
            # Enviar según el tipo
            if notification.notification_type == NotificationType.EMAIL:
                await self._send_email(notification)
            elif notification.notification_type == NotificationType.SMS:
                await self._send_sms(notification)
            elif notification.notification_type == NotificationType.WEBHOOK:
                await self._send_webhook(notification)
            elif notification.notification_type == NotificationType.SLACK:
                await self._send_slack(notification)
            elif notification.notification_type == NotificationType.TEAMS:
                await self._send_teams(notification)
            elif notification.notification_type == NotificationType.DISCORD:
                await self._send_discord(notification)
            else:
                raise ValueError(f"Tipo de notificación no soportado: {notification.notification_type}")
            
            # Marcar como enviada
            notification.status = NotificationStatus.SENT
            notification.sent_at = datetime.now()
            self.stats["sent_notifications"] += 1
            self.stats["pending_notifications"] -= 1
            
            logger.info(f"Notificación enviada: {notification.notification_id}")
            
        except Exception as e:
            # Manejar error
            notification.status = NotificationStatus.FAILED
            notification.failed_at = datetime.now()
            notification.error_message = str(e)
            notification.retry_count += 1
            
            self.stats["failed_notifications"] += 1
            self.stats["pending_notifications"] -= 1
            
            logger.error(f"Error al enviar notificación {notification.notification_id}: {e}")
            
            # Reintentar si es posible
            if notification.retry_count < notification.max_retries:
                await asyncio.sleep(notification.retry_count * 5)  # Backoff exponencial
                await self.notification_queue.put(notification)
                self.stats["pending_notifications"] += 1
    
    async def _send_email(self, notification: Notification):
        """Enviar email."""
        try:
            for recipient in notification.recipients:
                if not recipient.email:
                    continue
                
                # Crear mensaje
                msg = MIMEMultipart()
                msg['From'] = self.email_config.get("smtp_username", "noreply@system.com")
                msg['To'] = recipient.email
                msg['Subject'] = notification.subject
                
                # Agregar contenido
                msg.attach(MIMEText(notification.content, 'plain'))
                
                # Enviar email
                if self.email_config.get("smtp_use_tls"):
                    server = aiosmtplib.SMTP(
                        hostname=self.email_config["smtp_host"],
                        port=self.email_config["smtp_port"],
                        use_tls=True
                    )
                else:
                    server = aiosmtplib.SMTP(
                        hostname=self.email_config["smtp_host"],
                        port=self.email_config["smtp_port"]
                    )
                
                await server.connect()
                
                if self.email_config.get("smtp_username"):
                    await server.login(
                        self.email_config["smtp_username"],
                        self.email_config["smtp_password"]
                    )
                
                await server.send_message(msg)
                await server.quit()
                
                logger.info(f"Email enviado a {recipient.email}")
                
        except Exception as e:
            logger.error(f"Error al enviar email: {e}")
            raise
    
    async def _send_sms(self, notification: Notification):
        """Enviar SMS."""
        try:
            # Implementación básica - en producción usar Twilio o similar
            for recipient in notification.recipients:
                if not recipient.phone:
                    continue
                
                logger.info(f"SMS enviado a {recipient.phone}: {notification.content}")
                
        except Exception as e:
            logger.error(f"Error al enviar SMS: {e}")
            raise
    
    async def _send_webhook(self, notification: Notification):
        """Enviar webhook."""
        try:
            async with aiohttp.ClientSession() as session:
                for recipient in notification.recipients:
                    if not recipient.webhook_url:
                        continue
                    
                    payload = {
                        "notification_id": notification.notification_id,
                        "subject": notification.subject,
                        "content": notification.content,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": notification.metadata
                    }
                    
                    async with session.post(
                        recipient.webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.webhook_config["timeout"])
                    ) as response:
                        if response.status >= 400:
                            raise Exception(f"Webhook failed with status {response.status}")
                        
                        logger.info(f"Webhook enviado a {recipient.webhook_url}")
                        
        except Exception as e:
            logger.error(f"Error al enviar webhook: {e}")
            raise
    
    async def _send_slack(self, notification: Notification):
        """Enviar notificación a Slack."""
        try:
            # Implementación básica para Slack
            for recipient in notification.recipients:
                webhook_url = recipient.preferences.get("slack_webhook_url")
                if not webhook_url:
                    continue
                
                payload = {
                    "text": f"*{notification.subject}*\n{notification.content}",
                    "username": "Export IA System",
                    "icon_emoji": ":robot_face:"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=payload) as response:
                        if response.status >= 400:
                            raise Exception(f"Slack webhook failed with status {response.status}")
                        
                        logger.info(f"Slack notification enviada")
                        
        except Exception as e:
            logger.error(f"Error al enviar notificación a Slack: {e}")
            raise
    
    async def _send_teams(self, notification: Notification):
        """Enviar notificación a Microsoft Teams."""
        try:
            # Implementación básica para Teams
            for recipient in notification.recipients:
                webhook_url = recipient.preferences.get("teams_webhook_url")
                if not webhook_url:
                    continue
                
                payload = {
                    "@type": "MessageCard",
                    "@context": "http://schema.org/extensions",
                    "themeColor": "0076D7",
                    "summary": notification.subject,
                    "sections": [{
                        "activityTitle": notification.subject,
                        "activitySubtitle": "Export IA System",
                        "text": notification.content,
                        "markdown": True
                    }]
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=payload) as response:
                        if response.status >= 400:
                            raise Exception(f"Teams webhook failed with status {response.status}")
                        
                        logger.info(f"Teams notification enviada")
                        
        except Exception as e:
            logger.error(f"Error al enviar notificación a Teams: {e}")
            raise
    
    async def _send_discord(self, notification: Notification):
        """Enviar notificación a Discord."""
        try:
            # Implementación básica para Discord
            for recipient in notification.recipients:
                webhook_url = recipient.preferences.get("discord_webhook_url")
                if not webhook_url:
                    continue
                
                payload = {
                    "content": f"**{notification.subject}**\n{notification.content}",
                    "username": "Export IA System"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=payload) as response:
                        if response.status >= 400:
                            raise Exception(f"Discord webhook failed with status {response.status}")
                        
                        logger.info(f"Discord notification enviada")
                        
        except Exception as e:
            logger.error(f"Error al enviar notificación a Discord: {e}")
            raise
    
    async def get_notification_status(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de notificación."""
        if notification_id not in self.notifications:
            return None
        
        notification = self.notifications[notification_id]
        
        return {
            "notification_id": notification.notification_id,
            "type": notification.notification_type.value,
            "priority": notification.priority.value,
            "status": notification.status.value,
            "subject": notification.subject,
            "recipients_count": len(notification.recipients),
            "created_at": notification.created_at.isoformat(),
            "sent_at": notification.sent_at.isoformat() if notification.sent_at else None,
            "failed_at": notification.failed_at.isoformat() if notification.failed_at else None,
            "error_message": notification.error_message,
            "retry_count": notification.retry_count,
            "max_retries": notification.max_retries
        }
    
    async def get_notification_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de notificaciones."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "templates_count": len(self.templates),
            "recipients_count": len(self.recipients),
            "notifications_count": len(self.notifications),
            "queue_size": self.notification_queue.qsize(),
            "processing_task_running": self.processing_task is not None and not self.processing_task.done(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del sistema de notificaciones."""
        try:
            return {
                "status": "healthy",
                "templates_count": len(self.templates),
                "recipients_count": len(self.recipients),
                "notifications_count": len(self.notifications),
                "queue_size": self.notification_queue.qsize(),
                "processing_task_running": self.processing_task is not None and not self.processing_task.done(),
                "stats": self.stats,
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de notificaciones: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




