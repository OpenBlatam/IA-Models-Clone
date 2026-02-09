"""
Servicio de Notificaciones
==========================

Servicio para enviar notificaciones sobre el estado del procesamiento de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import smtplib
import json
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders

# Importaciones para notificaciones web
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Importaciones para Slack
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

# Importaciones para Discord
try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

logger = logging.getLogger(__name__)

class NotificationType(str, Enum):
    """Tipos de notificación"""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    CONSOLE = "console"

class NotificationPriority(str, Enum):
    """Prioridades de notificación"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class NotificationMessage:
    """Mensaje de notificación"""
    title: str
    content: str
    priority: NotificationPriority
    notification_type: NotificationType
    recipient: str
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class NotificationConfig:
    """Configuración de notificaciones"""
    enabled: bool = True
    email_config: Dict[str, Any] = None
    slack_config: Dict[str, Any] = None
    discord_config: Dict[str, Any] = None
    webhook_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.email_config is None:
            self.email_config = {}
        if self.slack_config is None:
            self.slack_config = {}
        if self.discord_config is None:
            self.discord_config = {}
        if self.webhook_config is None:
            self.webhook_config = {}

class NotificationService:
    """Servicio de notificaciones"""
    
    def __init__(self):
        self.config = NotificationConfig()
        self.slack_client = None
        self.discord_client = None
        self.notification_queue = asyncio.Queue()
        self.is_running = False
        
    async def initialize(self):
        """Inicializa el servicio de notificaciones"""
        logger.info("Inicializando servicio de notificaciones...")
        
        # Cargar configuración desde variables de entorno
        await self._load_configuration()
        
        # Inicializar clientes de notificación
        await self._initialize_clients()
        
        # Iniciar worker de notificaciones
        if self.config.enabled:
            asyncio.create_task(self._notification_worker())
            self.is_running = True
        
        logger.info("Servicio de notificaciones inicializado")
    
    async def _load_configuration(self):
        """Carga configuración desde variables de entorno"""
        import os
        
        # Configuración de email
        if os.getenv("NOTIFICATION_EMAIL_ENABLED", "false").lower() == "true":
            self.config.email_config = {
                "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("SMTP_USERNAME"),
                "password": os.getenv("SMTP_PASSWORD"),
                "from_email": os.getenv("FROM_EMAIL"),
                "to_emails": os.getenv("TO_EMAILS", "").split(",")
            }
        
        # Configuración de Slack
        if os.getenv("NOTIFICATION_SLACK_ENABLED", "false").lower() == "true":
            self.config.slack_config = {
                "token": os.getenv("SLACK_BOT_TOKEN"),
                "channel": os.getenv("SLACK_CHANNEL", "#general"),
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL")
            }
        
        # Configuración de Discord
        if os.getenv("NOTIFICATION_DISCORD_ENABLED", "false").lower() == "true":
            self.config.discord_config = {
                "bot_token": os.getenv("DISCORD_BOT_TOKEN"),
                "channel_id": os.getenv("DISCORD_CHANNEL_ID"),
                "webhook_url": os.getenv("DISCORD_WEBHOOK_URL")
            }
        
        # Configuración de Webhook
        if os.getenv("NOTIFICATION_WEBHOOK_ENABLED", "false").lower() == "true":
            self.config.webhook_config = {
                "url": os.getenv("WEBHOOK_URL"),
                "headers": json.loads(os.getenv("WEBHOOK_HEADERS", "{}")),
                "timeout": int(os.getenv("WEBHOOK_TIMEOUT", "30"))
            }
    
    async def _initialize_clients(self):
        """Inicializa clientes de notificación"""
        try:
            # Inicializar Slack
            if self.config.slack_config.get("token") and SLACK_AVAILABLE:
                self.slack_client = WebClient(token=self.config.slack_config["token"])
                logger.info("✅ Cliente Slack inicializado")
            
            # Inicializar Discord
            if self.config.discord_config.get("bot_token") and DISCORD_AVAILABLE:
                intents = discord.Intents.default()
                self.discord_client = discord.Client(intents=intents)
                logger.info("✅ Cliente Discord inicializado")
            
        except Exception as e:
            logger.warning(f"Error inicializando clientes de notificación: {e}")
    
    async def _notification_worker(self):
        """Worker que procesa la cola de notificaciones"""
        logger.info("Worker de notificaciones iniciado")
        
        while self.is_running:
            try:
                # Esperar notificación de la cola
                message = await asyncio.wait_for(
                    self.notification_queue.get(), 
                    timeout=1.0
                )
                
                # Procesar notificación
                await self._process_notification(message)
                
                # Marcar tarea como completada
                self.notification_queue.task_done()
                
            except asyncio.TimeoutError:
                # Timeout normal, continuar
                continue
            except Exception as e:
                logger.error(f"Error en worker de notificaciones: {e}")
                await asyncio.sleep(1)
    
    async def _process_notification(self, message: NotificationMessage):
        """Procesa una notificación individual"""
        try:
            if message.notification_type == NotificationType.EMAIL:
                await self._send_email_notification(message)
            elif message.notification_type == NotificationType.SLACK:
                await self._send_slack_notification(message)
            elif message.notification_type == NotificationType.DISCORD:
                await self._send_discord_notification(message)
            elif message.notification_type == NotificationType.WEBHOOK:
                await self._send_webhook_notification(message)
            elif message.notification_type == NotificationType.CONSOLE:
                await self._send_console_notification(message)
            
            logger.info(f"Notificación enviada: {message.title}")
            
        except Exception as e:
            logger.error(f"Error procesando notificación: {e}")
    
    async def _send_email_notification(self, message: NotificationMessage):
        """Envía notificación por email"""
        try:
            if not self.config.email_config:
                logger.warning("Configuración de email no disponible")
                return
            
            # Crear mensaje
            msg = MimeMultipart()
            msg['From'] = self.config.email_config["from_email"]
            msg['To'] = message.recipient
            msg['Subject'] = f"[{message.priority.value.upper()}] {message.title}"
            
            # Contenido del mensaje
            body = f"""
            {message.content}
            
            ---
            Prioridad: {message.priority.value}
            Timestamp: {message.timestamp.isoformat()}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Enviar email
            server = smtplib.SMTP(
                self.config.email_config["smtp_server"],
                self.config.email_config["smtp_port"]
            )
            server.starttls()
            server.login(
                self.config.email_config["username"],
                self.config.email_config["password"]
            )
            
            text = msg.as_string()
            server.sendmail(
                self.config.email_config["from_email"],
                message.recipient,
                text
            )
            server.quit()
            
        except Exception as e:
            logger.error(f"Error enviando email: {e}")
    
    async def _send_slack_notification(self, message: NotificationMessage):
        """Envía notificación a Slack"""
        try:
            if not self.slack_client and not self.config.slack_config.get("webhook_url"):
                logger.warning("Cliente Slack no disponible")
                return
            
            # Formatear mensaje
            color = {
                NotificationPriority.LOW: "#36a64f",
                NotificationPriority.MEDIUM: "#ffaa00",
                NotificationPriority.HIGH: "#ff6600",
                NotificationPriority.CRITICAL: "#ff0000"
            }.get(message.priority, "#36a64f")
            
            payload = {
                "channel": self.config.slack_config.get("channel", "#general"),
                "attachments": [
                    {
                        "color": color,
                        "title": message.title,
                        "text": message.content,
                        "fields": [
                            {
                                "title": "Prioridad",
                                "value": message.priority.value,
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": message.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            }
                        ],
                        "footer": "AI Document Processor",
                        "ts": int(message.timestamp.timestamp())
                    }
                ]
            }
            
            # Enviar usando webhook o cliente
            if self.config.slack_config.get("webhook_url"):
                if REQUESTS_AVAILABLE:
                    response = requests.post(
                        self.config.slack_config["webhook_url"],
                        json=payload,
                        timeout=10
                    )
                    response.raise_for_status()
            else:
                self.slack_client.chat_postMessage(
                    channel=self.config.slack_config["channel"],
                    text=message.title,
                    attachments=payload["attachments"]
                )
            
        except Exception as e:
            logger.error(f"Error enviando notificación a Slack: {e}")
    
    async def _send_discord_notification(self, message: NotificationMessage):
        """Envía notificación a Discord"""
        try:
            if not self.discord_client and not self.config.discord_config.get("webhook_url"):
                logger.warning("Cliente Discord no disponible")
                return
            
            # Formatear mensaje
            color = {
                NotificationPriority.LOW: 0x36a64f,
                NotificationPriority.MEDIUM: 0xffaa00,
                NotificationPriority.HIGH: 0xff6600,
                NotificationPriority.CRITICAL: 0xff0000
            }.get(message.priority, 0x36a64f)
            
            embed = {
                "title": message.title,
                "description": message.content,
                "color": color,
                "timestamp": message.timestamp.isoformat(),
                "footer": {
                    "text": "AI Document Processor"
                },
                "fields": [
                    {
                        "name": "Prioridad",
                        "value": message.priority.value,
                        "inline": True
                    }
                ]
            }
            
            # Enviar usando webhook o cliente
            if self.config.discord_config.get("webhook_url"):
                if REQUESTS_AVAILABLE:
                    payload = {"embeds": [embed]}
                    response = requests.post(
                        self.config.discord_config["webhook_url"],
                        json=payload,
                        timeout=10
                    )
                    response.raise_for_status()
            else:
                # Implementar envío con cliente Discord
                pass
            
        except Exception as e:
            logger.error(f"Error enviando notificación a Discord: {e}")
    
    async def _send_webhook_notification(self, message: NotificationMessage):
        """Envía notificación por webhook"""
        try:
            if not self.config.webhook_config.get("url"):
                logger.warning("URL de webhook no configurada")
                return
            
            if not REQUESTS_AVAILABLE:
                logger.warning("Biblioteca requests no disponible")
                return
            
            payload = {
                "title": message.title,
                "content": message.content,
                "priority": message.priority.value,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata
            }
            
            response = requests.post(
                self.config.webhook_config["url"],
                json=payload,
                headers=self.config.webhook_config.get("headers", {}),
                timeout=self.config.webhook_config.get("timeout", 30)
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error enviando webhook: {e}")
    
    async def _send_console_notification(self, message: NotificationMessage):
        """Envía notificación a consola"""
        try:
            priority_emoji = {
                NotificationPriority.LOW: "ℹ️",
                NotificationPriority.MEDIUM: "⚠️",
                NotificationPriority.HIGH: "🚨",
                NotificationPriority.CRITICAL: "🔥"
            }.get(message.priority, "ℹ️")
            
            print(f"\n{priority_emoji} NOTIFICACIÓN [{message.priority.value.upper()}]")
            print(f"📋 {message.title}")
            print(f"📝 {message.content}")
            print(f"⏰ {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 50)
            
        except Exception as e:
            logger.error(f"Error enviando notificación a consola: {e}")
    
    async def send_notification(
        self,
        title: str,
        content: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        notification_type: NotificationType = NotificationType.CONSOLE,
        recipient: str = "",
        metadata: Dict[str, Any] = None
    ):
        """Envía una notificación"""
        try:
            if not self.config.enabled:
                logger.debug("Notificaciones deshabilitadas")
                return
            
            message = NotificationMessage(
                title=title,
                content=content,
                priority=priority,
                notification_type=notification_type,
                recipient=recipient,
                metadata=metadata or {}
            )
            
            # Agregar a la cola
            await self.notification_queue.put(message)
            
        except Exception as e:
            logger.error(f"Error enviando notificación: {e}")
    
    async def notify_document_processed(
        self,
        filename: str,
        success: bool,
        processing_time: float,
        target_format: str,
        error_message: str = None
    ):
        """Notifica el procesamiento de un documento"""
        try:
            if success:
                title = f"✅ Documento procesado exitosamente"
                content = f"""
                Archivo: {filename}
                Formato objetivo: {target_format}
                Tiempo de procesamiento: {processing_time:.2f} segundos
                """
                priority = NotificationPriority.LOW
            else:
                title = f"❌ Error procesando documento"
                content = f"""
                Archivo: {filename}
                Formato objetivo: {target_format}
                Error: {error_message or 'Error desconocido'}
                Tiempo de procesamiento: {processing_time:.2f} segundos
                """
                priority = NotificationPriority.HIGH
            
            await self.send_notification(
                title=title,
                content=content,
                priority=priority,
                notification_type=NotificationType.CONSOLE,
                metadata={
                    "filename": filename,
                    "success": success,
                    "processing_time": processing_time,
                    "target_format": target_format
                }
            )
            
        except Exception as e:
            logger.error(f"Error notificando procesamiento de documento: {e}")
    
    async def notify_batch_completed(
        self,
        total_files: int,
        successful: int,
        failed: int,
        processing_time: float
    ):
        """Notifica la finalización de un lote"""
        try:
            success_rate = (successful / total_files * 100) if total_files > 0 else 0
            
            if failed == 0:
                title = f"🎉 Lote completado exitosamente"
                priority = NotificationPriority.LOW
            elif failed < total_files / 2:
                title = f"⚠️ Lote completado con algunos errores"
                priority = NotificationPriority.MEDIUM
            else:
                title = f"🚨 Lote completado con muchos errores"
                priority = NotificationPriority.HIGH
            
            content = f"""
            Total de archivos: {total_files}
            Exitosos: {successful}
            Fallidos: {failed}
            Tasa de éxito: {success_rate:.1f}%
            Tiempo total: {processing_time:.2f} segundos
            """
            
            await self.send_notification(
                title=title,
                content=content,
                priority=priority,
                notification_type=NotificationType.CONSOLE,
                metadata={
                    "total_files": total_files,
                    "successful": successful,
                    "failed": failed,
                    "processing_time": processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error notificando finalización de lote: {e}")
    
    async def notify_system_error(
        self,
        error_type: str,
        error_message: str,
        component: str = "unknown"
    ):
        """Notifica un error del sistema"""
        try:
            title = f"🚨 Error del sistema: {error_type}"
            content = f"""
            Componente: {component}
            Error: {error_message}
            Timestamp: {datetime.now().isoformat()}
            """
            
            await self.send_notification(
                title=title,
                content=content,
                priority=NotificationPriority.CRITICAL,
                notification_type=NotificationType.CONSOLE,
                metadata={
                    "error_type": error_type,
                    "component": component,
                    "error_message": error_message
                }
            )
            
        except Exception as e:
            logger.error(f"Error notificando error del sistema: {e}")
    
    async def stop(self):
        """Detiene el servicio de notificaciones"""
        try:
            self.is_running = False
            
            # Esperar a que se procesen las notificaciones pendientes
            await self.notification_queue.join()
            
            logger.info("Servicio de notificaciones detenido")
            
        except Exception as e:
            logger.error(f"Error deteniendo servicio de notificaciones: {e}")


