"""
Advanced Notification System for AI History Comparison
Sistema avanzado de notificaciones para el sistema de análisis de historial de IA
"""

import asyncio
import json
import logging
import smtplib
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import aiohttp
import redis
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationChannel(Enum):
    """Canales de notificación disponibles"""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    SMS = "sms"

class NotificationPriority(Enum):
    """Prioridades de notificación"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class NotificationStatus(Enum):
    """Estados de notificación"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DELIVERED = "delivered"
    READ = "read"

@dataclass
class NotificationTemplate:
    """Plantilla de notificación"""
    id: str
    name: str
    subject: str
    body: str
    channels: List[NotificationChannel]
    priority: NotificationPriority
    variables: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class NotificationRecipient:
    """Destinatario de notificación"""
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_user: Optional[str] = None
    discord_user: Optional[str] = None
    telegram_user: Optional[str] = None
    webhook_url: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

@dataclass
class Notification:
    """Notificación individual"""
    id: str
    template_id: str
    recipient_id: str
    channel: NotificationChannel
    priority: NotificationPriority
    subject: str
    body: str
    variables: Dict[str, Any] = field(default_factory=dict)
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None

class NotificationSystem:
    """
    Sistema avanzado de notificaciones para el sistema de análisis de historial de IA
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
        discord_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        sms_provider_config: Optional[Dict[str, Any]] = None
    ):
        self.redis_url = redis_url
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.slack_webhook_url = slack_webhook_url
        self.discord_webhook_url = discord_webhook_url
        self.telegram_bot_token = telegram_bot_token
        self.sms_provider_config = sms_provider_config or {}
        
        # Initialize Redis
        self.redis_client = redis.from_url(redis_url)
        
        # Storage
        self.templates: Dict[str, NotificationTemplate] = {}
        self.recipients: Dict[str, NotificationRecipient] = {}
        self.notifications: Dict[str, Notification] = {}
        
        # Callbacks
        self.channel_callbacks: Dict[NotificationChannel, Callable] = {}
        
        # Statistics
        self.stats = {
            "total_sent": 0,
            "total_delivered": 0,
            "total_failed": 0,
            "by_channel": {},
            "by_priority": {},
            "by_status": {}
        }
        
        # Initialize default templates
        self._initialize_default_templates()
        
        # Setup channel callbacks
        self._setup_channel_callbacks()
    
    def _initialize_default_templates(self):
        """Inicializar plantillas por defecto"""
        
        # Template para alertas críticas
        self.templates["critical_alert"] = NotificationTemplate(
            id="critical_alert",
            name="Alerta Crítica",
            subject="🚨 Alerta Crítica - Sistema AI History",
            body="""
            <h2>🚨 Alerta Crítica Detectada</h2>
            <p><strong>Tipo:</strong> {alert_type}</p>
            <p><strong>Mensaje:</strong> {message}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Valor Actual:</strong> {current_value}</p>
            <p><strong>Umbral:</strong> {threshold}</p>
            <p><strong>Documento ID:</strong> {document_id}</p>
            
            <p>Por favor, revise el sistema inmediatamente.</p>
            
            <p>Saludos,<br>Sistema de Monitoreo AI History</p>
            """,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            priority=NotificationPriority.CRITICAL,
            variables=["alert_type", "message", "timestamp", "current_value", "threshold", "document_id"]
        )
        
        # Template para alertas de advertencia
        self.templates["warning_alert"] = NotificationTemplate(
            id="warning_alert",
            name="Alerta de Advertencia",
            subject="⚠️ Alerta de Advertencia - Sistema AI History",
            body="""
            <h2>⚠️ Alerta de Advertencia</h2>
            <p><strong>Tipo:</strong> {alert_type}</p>
            <p><strong>Mensaje:</strong> {message}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Valor Actual:</strong> {current_value}</p>
            <p><strong>Umbral:</strong> {threshold}</p>
            
            <p>Se recomienda revisar el sistema.</p>
            
            <p>Saludos,<br>Sistema de Monitoreo AI History</p>
            """,
            channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD],
            priority=NotificationPriority.MEDIUM,
            variables=["alert_type", "message", "timestamp", "current_value", "threshold"]
        )
        
        # Template para insights importantes
        self.templates["important_insight"] = NotificationTemplate(
            id="important_insight",
            name="Insight Importante",
            subject="💡 Insight Importante - Sistema AI History",
            body="""
            <h2>💡 Nuevo Insight Importante</h2>
            <p><strong>Título:</strong> {title}</p>
            <p><strong>Descripción:</strong> {description}</p>
            <p><strong>Confianza:</strong> {confidence}%</p>
            <p><strong>Impacto:</strong> {impact_score}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            
            <h3>Recomendaciones:</h3>
            <ul>
            {recommendations}
            </ul>
            
            <p>Saludos,<br>Sistema de Análisis AI History</p>
            """,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            priority=NotificationPriority.MEDIUM,
            variables=["title", "description", "confidence", "impact_score", "timestamp", "recommendations"]
        )
        
        # Template para optimización ML completada
        self.templates["ml_optimization_complete"] = NotificationTemplate(
            id="ml_optimization_complete",
            name="Optimización ML Completada",
            subject="🤖 Optimización ML Completada - Sistema AI History",
            body="""
            <h2>🤖 Optimización de ML Completada</h2>
            <p><strong>Mejor Modelo:</strong> {best_model}</p>
            <p><strong>Puntuación R²:</strong> {best_score}</p>
            <p><strong>Mejora:</strong> {improvement}%</p>
            <p><strong>Documentos Analizados:</strong> {documents_count}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            
            <h3>Recomendaciones:</h3>
            <ul>
            {recommendations}
            </ul>
            
            <p>El modelo ha sido actualizado y está listo para uso.</p>
            
            <p>Saludos,<br>Sistema de ML AI History</p>
            """,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            priority=NotificationPriority.LOW,
            variables=["best_model", "best_score", "improvement", "documents_count", "timestamp", "recommendations"]
        )
        
        # Template para reporte diario
        self.templates["daily_report"] = NotificationTemplate(
            id="daily_report",
            name="Reporte Diario",
            subject="📊 Reporte Diario - Sistema AI History",
            body="""
            <h2>📊 Reporte Diario del Sistema</h2>
            <p><strong>Fecha:</strong> {date}</p>
            
            <h3>Estadísticas del Día:</h3>
            <ul>
                <li><strong>Documentos Procesados:</strong> {total_documents}</li>
                <li><strong>Calidad Promedio:</strong> {avg_quality}</li>
                <li><strong>Alertas Generadas:</strong> {total_alerts}</li>
                <li><strong>Insights Generados:</strong> {total_insights}</li>
            </ul>
            
            <h3>Mejores Documentos:</h3>
            {best_documents}
            
            <h3>Alertas Activas:</h3>
            {active_alerts}
            
            <p>Saludos,<br>Sistema de Monitoreo AI History</p>
            """,
            channels=[NotificationChannel.EMAIL],
            priority=NotificationPriority.LOW,
            variables=["date", "total_documents", "avg_quality", "total_alerts", "total_insights", "best_documents", "active_alerts"]
        )
    
    def _setup_channel_callbacks(self):
        """Configurar callbacks para cada canal"""
        self.channel_callbacks = {
            NotificationChannel.EMAIL: self._send_email,
            NotificationChannel.SLACK: self._send_slack,
            NotificationChannel.DISCORD: self._send_discord,
            NotificationChannel.TELEGRAM: self._send_telegram,
            NotificationChannel.WEBHOOK: self._send_webhook,
            NotificationChannel.DASHBOARD: self._send_dashboard,
            NotificationChannel.SMS: self._send_sms
        }
    
    async def add_recipient(self, recipient: NotificationRecipient):
        """Agregar destinatario"""
        self.recipients[recipient.id] = recipient
        logger.info(f"Recipient added: {recipient.id}")
    
    async def add_template(self, template: NotificationTemplate):
        """Agregar plantilla"""
        self.templates[template.id] = template
        logger.info(f"Template added: {template.id}")
    
    async def send_notification(
        self,
        template_id: str,
        recipient_id: str,
        variables: Dict[str, Any],
        channels: Optional[List[NotificationChannel]] = None,
        priority: Optional[NotificationPriority] = None
    ) -> List[str]:
        """
        Enviar notificación usando una plantilla
        
        Args:
            template_id: ID de la plantilla
            recipient_id: ID del destinatario
            variables: Variables para reemplazar en la plantilla
            channels: Canales específicos (opcional)
            priority: Prioridad específica (opcional)
            
        Returns:
            Lista de IDs de notificaciones enviadas
        """
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        if recipient_id not in self.recipients:
            raise ValueError(f"Recipient {recipient_id} not found")
        
        template = self.templates[template_id]
        recipient = self.recipients[recipient_id]
        
        # Usar canales de la plantilla si no se especifican
        if channels is None:
            channels = template.channels
        
        # Usar prioridad de la plantilla si no se especifica
        if priority is None:
            priority = template.priority
        
        # Procesar plantilla
        subject = self._process_template(template.subject, variables)
        body = self._process_template(template.body, variables)
        
        notification_ids = []
        
        # Enviar por cada canal
        for channel in channels:
            notification = Notification(
                id=f"{template_id}_{recipient_id}_{channel.value}_{datetime.now().timestamp()}",
                template_id=template_id,
                recipient_id=recipient_id,
                channel=channel,
                priority=priority,
                subject=subject,
                body=body,
                variables=variables
            )
            
            # Enviar notificación
            success = await self._send_notification(notification)
            
            if success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now()
                self.stats["total_sent"] += 1
            else:
                notification.status = NotificationStatus.FAILED
                self.stats["total_failed"] += 1
            
            # Almacenar notificación
            self.notifications[notification.id] = notification
            notification_ids.append(notification.id)
            
            # Actualizar estadísticas
            self._update_stats(notification)
        
        return notification_ids
    
    async def _send_notification(self, notification: Notification) -> bool:
        """Enviar notificación individual"""
        try:
            if notification.channel in self.channel_callbacks:
                callback = self.channel_callbacks[notification.channel]
                success = await callback(notification)
                
                if success:
                    notification.delivered_at = datetime.now()
                    self.stats["total_delivered"] += 1
                
                return success
            else:
                logger.error(f"No callback found for channel: {notification.channel}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending notification {notification.id}: {e}")
            notification.error_message = str(e)
            notification.retry_count += 1
            
            # Reintentar si no se ha alcanzado el máximo
            if notification.retry_count < notification.max_retries:
                await asyncio.sleep(60 * notification.retry_count)  # Backoff exponencial
                return await self._send_notification(notification)
            
            return False
    
    def _process_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Procesar plantilla reemplazando variables"""
        processed = template
        
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if isinstance(value, list):
                # Procesar listas como HTML
                if key == "recommendations":
                    list_items = "".join([f"<li>{item}</li>" for item in value])
                    processed = processed.replace(placeholder, list_items)
                else:
                    processed = processed.replace(placeholder, ", ".join(map(str, value)))
            else:
                processed = processed.replace(placeholder, str(value))
        
        return processed
    
    async def _send_email(self, notification: Notification) -> bool:
        """Enviar notificación por email"""
        try:
            if not self.smtp_server or not self.smtp_username or not self.smtp_password:
                logger.warning("SMTP not configured, skipping email notification")
                return False
            
            recipient = self.recipients[notification.recipient_id]
            if not recipient.email:
                logger.warning(f"No email address for recipient {recipient.id}")
                return False
            
            # Crear mensaje
            msg = MIMEMultipart('alternative')
            msg['From'] = self.smtp_username
            msg['To'] = recipient.email
            msg['Subject'] = notification.subject
            
            # Agregar cuerpo HTML
            html_body = MIMEText(notification.body, 'html', 'utf-8')
            msg.attach(html_body)
            
            # Enviar email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {recipient.email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_slack(self, notification: Notification) -> bool:
        """Enviar notificación a Slack"""
        try:
            if not self.slack_webhook_url:
                logger.warning("Slack webhook not configured")
                return False
            
            # Crear payload para Slack
            payload = {
                "text": f"*{notification.subject}*",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": notification.body.replace('<h2>', '*').replace('</h2>', '*').replace('<p>', '').replace('</p>', '\n').replace('<strong>', '*').replace('</strong>', '*')
                        }
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack notification sent")
                        return True
                    else:
                        logger.error(f"Slack API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_discord(self, notification: Notification) -> bool:
        """Enviar notificación a Discord"""
        try:
            if not self.discord_webhook_url:
                logger.warning("Discord webhook not configured")
                return False
            
            # Crear payload para Discord
            payload = {
                "embeds": [
                    {
                        "title": notification.subject,
                        "description": notification.body.replace('<h2>', '').replace('</h2>', '').replace('<p>', '').replace('</p>', '\n').replace('<strong>', '**').replace('</strong>', '**'),
                        "color": self._get_priority_color(notification.priority),
                        "timestamp": notification.created_at.isoformat()
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook_url, json=payload) as response:
                    if response.status == 204:
                        logger.info("Discord notification sent")
                        return True
                    else:
                        logger.error(f"Discord API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False
    
    async def _send_telegram(self, notification: Notification) -> bool:
        """Enviar notificación a Telegram"""
        try:
            if not self.telegram_bot_token:
                logger.warning("Telegram bot token not configured")
                return False
            
            recipient = self.recipients[notification.recipient_id]
            if not recipient.telegram_user:
                logger.warning(f"No Telegram user for recipient {recipient.id}")
                return False
            
            # Crear mensaje
            message = f"*{notification.subject}*\n\n{notification.body.replace('<h2>', '').replace('</h2>', '').replace('<p>', '').replace('</p>', '\n').replace('<strong>', '*').replace('</strong>', '*')}"
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": recipient.telegram_user,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Telegram notification sent")
                        return True
                    else:
                        logger.error(f"Telegram API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False
    
    async def _send_webhook(self, notification: Notification) -> bool:
        """Enviar notificación a webhook personalizado"""
        try:
            recipient = self.recipients[notification.recipient_id]
            if not recipient.webhook_url:
                logger.warning(f"No webhook URL for recipient {recipient.id}")
                return False
            
            payload = {
                "notification_id": notification.id,
                "template_id": notification.template_id,
                "subject": notification.subject,
                "body": notification.body,
                "priority": notification.priority.value,
                "timestamp": notification.created_at.isoformat(),
                "variables": notification.variables
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(recipient.webhook_url, json=payload) as response:
                    if response.status in [200, 201, 202]:
                        logger.info("Webhook notification sent")
                        return True
                    else:
                        logger.error(f"Webhook error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    async def _send_dashboard(self, notification: Notification) -> bool:
        """Enviar notificación al dashboard"""
        try:
            # Almacenar en Redis para que el dashboard lo lea
            notification_data = {
                "id": notification.id,
                "subject": notification.subject,
                "body": notification.body,
                "priority": notification.priority.value,
                "timestamp": notification.created_at.isoformat()
            }
            
            self.redis_client.lpush("dashboard_notifications", json.dumps(notification_data))
            self.redis_client.expire("dashboard_notifications", 86400)  # 24 horas
            
            logger.info("Dashboard notification sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending dashboard notification: {e}")
            return False
    
    async def _send_sms(self, notification: Notification) -> bool:
        """Enviar notificación por SMS"""
        try:
            recipient = self.recipients[notification.recipient_id]
            if not recipient.phone:
                logger.warning(f"No phone number for recipient {recipient.id}")
                return False
            
            # Implementar según el proveedor SMS configurado
            # Por ahora, solo log
            logger.info(f"SMS would be sent to {recipient.phone}: {notification.subject}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    def _get_priority_color(self, priority: NotificationPriority) -> int:
        """Obtener color para Discord según prioridad"""
        colors = {
            NotificationPriority.LOW: 0x00ff00,      # Verde
            NotificationPriority.MEDIUM: 0xffff00,   # Amarillo
            NotificationPriority.HIGH: 0xff8000,     # Naranja
            NotificationPriority.CRITICAL: 0xff0000, # Rojo
            NotificationPriority.EMERGENCY: 0x800080 # Morado
        }
        return colors.get(priority, 0x0000ff)  # Azul por defecto
    
    def _update_stats(self, notification: Notification):
        """Actualizar estadísticas"""
        # Por canal
        channel = notification.channel.value
        if channel not in self.stats["by_channel"]:
            self.stats["by_channel"][channel] = 0
        self.stats["by_channel"][channel] += 1
        
        # Por prioridad
        priority = notification.priority.value
        if priority not in self.stats["by_priority"]:
            self.stats["by_priority"][priority] = 0
        self.stats["by_priority"][priority] += 1
        
        # Por estado
        status = notification.status.value
        if status not in self.stats["by_status"]:
            self.stats["by_status"][status] = 0
        self.stats["by_status"][status] += 1
    
    async def send_bulk_notification(
        self,
        template_id: str,
        recipient_ids: List[str],
        variables: Dict[str, Any],
        channels: Optional[List[NotificationChannel]] = None
    ) -> Dict[str, List[str]]:
        """Enviar notificación masiva"""
        results = {}
        
        for recipient_id in recipient_ids:
            try:
                notification_ids = await self.send_notification(
                    template_id=template_id,
                    recipient_id=recipient_id,
                    variables=variables,
                    channels=channels
                )
                results[recipient_id] = notification_ids
            except Exception as e:
                logger.error(f"Error sending bulk notification to {recipient_id}: {e}")
                results[recipient_id] = []
        
        return results
    
    async def schedule_notification(
        self,
        template_id: str,
        recipient_id: str,
        variables: Dict[str, Any],
        scheduled_time: datetime,
        channels: Optional[List[NotificationChannel]] = None
    ) -> str:
        """Programar notificación para envío futuro"""
        notification_data = {
            "template_id": template_id,
            "recipient_id": recipient_id,
            "variables": variables,
            "channels": [c.value for c in (channels or [])],
            "scheduled_time": scheduled_time.isoformat()
        }
        
        # Almacenar en Redis con timestamp como score
        notification_id = f"scheduled_{datetime.now().timestamp()}"
        self.redis_client.zadd("scheduled_notifications", {
            json.dumps(notification_data): scheduled_time.timestamp()
        })
        
        logger.info(f"Notification scheduled for {scheduled_time}")
        return notification_id
    
    async def process_scheduled_notifications(self):
        """Procesar notificaciones programadas"""
        current_time = datetime.now().timestamp()
        
        # Obtener notificaciones que deben enviarse
        scheduled = self.redis_client.zrangebyscore(
            "scheduled_notifications", 
            0, 
            current_time
        )
        
        for notification_data in scheduled:
            try:
                data = json.loads(notification_data)
                
                # Enviar notificación
                await self.send_notification(
                    template_id=data["template_id"],
                    recipient_id=data["recipient_id"],
                    variables=data["variables"],
                    channels=[NotificationChannel(c) for c in data.get("channels", [])]
                )
                
                # Remover de la cola
                self.redis_client.zrem("scheduled_notifications", notification_data)
                
            except Exception as e:
                logger.error(f"Error processing scheduled notification: {e}")
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de notificaciones"""
        return {
            "total_templates": len(self.templates),
            "total_recipients": len(self.recipients),
            "total_notifications": len(self.notifications),
            "stats": self.stats,
            "recent_notifications": [
                {
                    "id": n.id,
                    "template_id": n.template_id,
                    "recipient_id": n.recipient_id,
                    "channel": n.channel.value,
                    "priority": n.priority.value,
                    "status": n.status.value,
                    "created_at": n.created_at.isoformat()
                }
                for n in sorted(
                    self.notifications.values(),
                    key=lambda x: x.created_at,
                    reverse=True
                )[:10]
            ]
        }
    
    async def cleanup_old_notifications(self, days: int = 30):
        """Limpiar notificaciones antiguas"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_notifications = [
            nid for nid, notification in self.notifications.items()
            if notification.created_at < cutoff_date
        ]
        
        for nid in old_notifications:
            del self.notifications[nid]
        
        logger.info(f"Cleaned up {len(old_notifications)} old notifications")
    
    async def start_scheduler(self):
        """Iniciar procesador de notificaciones programadas"""
        while True:
            try:
                await self.process_scheduled_notifications()
                await asyncio.sleep(60)  # Verificar cada minuto
            except Exception as e:
                logger.error(f"Error in notification scheduler: {e}")
                await asyncio.sleep(60)



























