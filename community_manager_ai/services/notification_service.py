"""
Notification Service - Servicio de Notificaciones
===================================================

Sistema para enviar notificaciones sobre eventos del sistema.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Tipos de notificaciones"""
    POST_PUBLISHED = "post_published"
    POST_FAILED = "post_failed"
    POST_SCHEDULED = "post_scheduled"
    PLATFORM_CONNECTED = "platform_connected"
    PLATFORM_DISCONNECTED = "platform_disconnected"
    ENGAGEMENT_MILESTONE = "engagement_milestone"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class NotificationService:
    """Servicio de notificaciones"""
    
    def __init__(self):
        """Inicializar servicio de notificaciones"""
        self.handlers: Dict[NotificationType, List[Callable]] = defaultdict(list)
        self.notifications: List[Dict[str, Any]] = []
        logger.info("Notification Service inicializado")
    
    def register_handler(
        self,
        notification_type: NotificationType,
        handler: Callable[[Dict[str, Any]], None]
    ):
        """
        Registrar un handler para un tipo de notificación
        
        Args:
            notification_type: Tipo de notificación
            handler: Función que maneja la notificación
        """
        self.handlers[notification_type].append(handler)
        logger.info(f"Handler registrado para {notification_type.value}")
    
    def send_notification(
        self,
        notification_type: NotificationType,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Enviar una notificación
        
        Args:
            notification_type: Tipo de notificación
            message: Mensaje de la notificación
            data: Datos adicionales
        """
        notification = {
            "type": notification_type.value,
            "message": message,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.notifications.append(notification)
        
        # Ejecutar handlers
        handlers = self.handlers.get(notification_type, [])
        for handler in handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Error en handler de notificación: {e}")
        
        logger.info(f"Notificación enviada: {notification_type.value} - {message}")
    
    def get_notifications(
        self,
        notification_type: Optional[NotificationType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener notificaciones
        
        Args:
            notification_type: Filtrar por tipo
            limit: Límite de notificaciones
            
        Returns:
            Lista de notificaciones
        """
        notifications = self.notifications
        
        if notification_type:
            notifications = [
                n for n in notifications
                if n.get("type") == notification_type.value
            ]
        
        # Ordenar por timestamp (más recientes primero)
        notifications.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return notifications[:limit]
    
    def clear_notifications(self, older_than_days: int = 30):
        """
        Limpiar notificaciones antiguas
        
        Args:
            older_than_days: Eliminar notificaciones más antiguas que X días
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        before_count = len(self.notifications)
        self.notifications = [
            n for n in self.notifications
            if datetime.fromisoformat(n.get("timestamp", "")) >= cutoff_date
        ]
        after_count = len(self.notifications)
        
        logger.info(f"Notificaciones limpiadas: {before_count - after_count} eliminadas (manteniendo últimas {older_than_days} días)")
    
    def get_unread_count(
        self,
        notification_type: Optional[NotificationType] = None
    ) -> int:
        """
        Obtener conteo de notificaciones no leídas
        
        Args:
            notification_type: Filtrar por tipo
            
        Returns:
            Número de notificaciones no leídas
        """
        notifications = self.notifications
        
        if notification_type:
            notifications = [
                n for n in notifications
                if n.get("type") == notification_type.value
            ]
        
        unread = [n for n in notifications if not n.get("read", False)]
        return len(unread)
    
    def mark_as_read(
        self,
        notification_id: Optional[str] = None,
        notification_type: Optional[NotificationType] = None
    ) -> int:
        """
        Marcar notificaciones como leídas
        
        Args:
            notification_id: ID específico de notificación
            notification_type: Marcar todas de un tipo
            
        Returns:
            Número de notificaciones marcadas
        """
        marked = 0
        
        for notification in self.notifications:
            if notification_id and notification.get("id") == notification_id:
                notification["read"] = True
                notification["read_at"] = datetime.now().isoformat()
                marked += 1
            elif notification_type and notification.get("type") == notification_type.value:
                if not notification.get("read", False):
                    notification["read"] = True
                    notification["read_at"] = datetime.now().isoformat()
                    marked += 1
        
        if marked > 0:
            logger.info(f"{marked} notificaciones marcadas como leídas")
        
        return marked
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de notificaciones
        
        Returns:
            Dict con estadísticas
        """
        total = len(self.notifications)
        
        by_type = defaultdict(int)
        by_status = {"read": 0, "unread": 0}
        
        for notification in self.notifications:
            notif_type = notification.get("type", "unknown")
            by_type[notif_type] += 1
            
            if notification.get("read", False):
                by_status["read"] += 1
            else:
                by_status["unread"] += 1
        
        return {
            "total": total,
            "by_type": dict(by_type),
            "by_status": by_status,
            "handlers_registered": sum(len(handlers) for handlers in self.handlers.values())
        }


# Handlers predefinidos
def log_notification_handler(notification: Dict[str, Any]):
    """Handler que solo loguea la notificación"""
    logger.info(f"NOTIFICATION [{notification['type']}]: {notification['message']}")


def email_notification_handler(notification: Dict[str, Any]):
    """Handler para enviar por email"""
    try:
        from ..config import get_settings
        settings = get_settings()
        
        email_enabled = getattr(settings, 'email_enabled', False)
        if not email_enabled:
            logger.debug("Email notifications disabled")
            return
        
        recipient = notification.get("data", {}).get("recipient")
        if not recipient:
            logger.warning("No recipient specified for email notification")
            return
        
        subject = f"[Community Manager AI] {notification.get('type', 'Notification').upper()}"
        message = notification.get("message", "")
        
        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            smtp_host = getattr(settings, 'smtp_host', 'localhost')
            smtp_port = getattr(settings, 'smtp_port', 587)
            smtp_user = getattr(settings, 'smtp_user', '')
            smtp_password = getattr(settings, 'smtp_password', '')
            smtp_from = getattr(settings, 'smtp_from', smtp_user)
            
            msg = MIMEMultipart()
            msg['From'] = smtp_from
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain'))
            
            async def send_email():
                await aiosmtplib.send(
                    msg,
                    hostname=smtp_host,
                    port=smtp_port,
                    username=smtp_user if smtp_user else None,
                    password=smtp_password if smtp_password else None,
                    use_tls=smtp_port == 587
                )
            
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(send_email())
                else:
                    loop.run_until_complete(send_email())
            except RuntimeError:
                asyncio.run(send_email())
            
            logger.info(f"Email notification sent to {recipient}")
        except ImportError:
            logger.warning("aiosmtplib not available, email notification skipped")
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    except Exception as e:
        logger.error(f"Error in email notification handler: {e}")


def webhook_notification_handler(notification: Dict[str, Any]):
    """Handler para enviar webhook"""
    try:
        from ..config import get_settings
        settings = get_settings()
        
        webhook_url = notification.get("data", {}).get("webhook_url")
        if not webhook_url:
            webhook_url = getattr(settings, 'default_webhook_url', None)
        
        if not webhook_url:
            logger.debug("No webhook URL specified, skipping webhook notification")
            return
        
        try:
            import httpx
            
            payload = {
                "type": notification.get("type"),
                "message": notification.get("message"),
                "data": notification.get("data", {}),
                "timestamp": notification.get("timestamp")
            }
            
            webhook_secret = getattr(settings, 'webhook_secret', None)
            headers = {"Content-Type": "application/json"}
            
            if webhook_secret:
                import hmac
                import hashlib
                import json
                signature = hmac.new(
                    webhook_secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Webhook-Signature"] = signature
            
            async def send_webhook():
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(webhook_url, json=payload, headers=headers)
                    response.raise_for_status()
            
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(send_webhook())
                else:
                    loop.run_until_complete(send_webhook())
            except RuntimeError:
                asyncio.run(send_webhook())
            
            logger.info(f"Webhook notification sent to {webhook_url}")
        except ImportError:
            logger.warning("httpx not available, webhook notification skipped")
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    except Exception as e:
        logger.error(f"Error in webhook notification handler: {e}")

