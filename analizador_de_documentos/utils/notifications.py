"""
Sistema de Notificaciones y Webhooks
=====================================

Sistema para enviar notificaciones y webhooks cuando se completan análisis.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import httpx
import json

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Tipos de notificación"""
    ANALYSIS_COMPLETE = "analysis_complete"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    TREND_ALERT = "trend_alert"


@dataclass
class Notification:
    """Notificación"""
    type: NotificationType
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class NotificationHandler:
    """Manejador de notificaciones"""
    
    def __init__(self):
        """Inicializar manejador"""
        self.handlers: List[Callable] = []
        self.webhooks: List[str] = []
        logger.info("NotificationHandler inicializado")
    
    def register_handler(self, handler: Callable):
        """Registrar handler de notificaciones"""
        self.handlers.append(handler)
    
    def register_webhook(self, url: str):
        """Registrar webhook URL"""
        if url not in self.webhooks:
            self.webhooks.append(url)
            logger.info(f"Webhook registrado: {url}")
    
    async def send_notification(self, notification: Notification):
        """Enviar notificación"""
        # Ejecutar handlers locales
        for handler in self.handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(notification)
                else:
                    handler(notification)
            except Exception as e:
                logger.error(f"Error en handler de notificación: {e}")
        
        # Enviar webhooks
        await self._send_webhooks(notification)
    
    async def _send_webhooks(self, notification: Notification):
        """Enviar webhooks"""
        payload = {
            "type": notification.type.value,
            "title": notification.title,
            "message": notification.message,
            "data": notification.data,
            "timestamp": notification.timestamp
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for webhook_url in self.webhooks:
                try:
                    response = await client.post(
                        webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    logger.debug(f"Webhook enviado a {webhook_url}: {response.status_code}")
                except Exception as e:
                    logger.error(f"Error enviando webhook a {webhook_url}: {e}")


class WebhookManager:
    """Gestor de webhooks"""
    
    def __init__(self):
        """Inicializar gestor"""
        self.notification_handler = NotificationHandler()
        logger.info("WebhookManager inicializado")
    
    async def notify_analysis_complete(
        self,
        document_id: str,
        analysis_result: Dict[str, Any],
        webhook_url: Optional[str] = None
    ):
        """Notificar análisis completado"""
        notification = Notification(
            type=NotificationType.ANALYSIS_COMPLETE,
            title="Análisis Completado",
            message=f"Análisis del documento {document_id} completado",
            data={
                "document_id": document_id,
                "analysis_result": analysis_result
            }
        )
        
        if webhook_url:
            self.notification_handler.register_webhook(webhook_url)
        
        await self.notification_handler.send_notification(notification)
    
    async def notify_error(
        self,
        error_message: str,
        error_data: Optional[Dict[str, Any]] = None,
        webhook_url: Optional[str] = None
    ):
        """Notificar error"""
        notification = Notification(
            type=NotificationType.ERROR,
            title="Error en Análisis",
            message=error_message,
            data=error_data or {}
        )
        
        if webhook_url:
            self.notification_handler.register_webhook(webhook_url)
        
        await self.notification_handler.send_notification(notification)
    
    async def notify_trend_alert(
        self,
        trend_name: str,
        alert_message: str,
        trend_data: Dict[str, Any],
        webhook_url: Optional[str] = None
    ):
        """Notificar alerta de tendencia"""
        notification = Notification(
            type=NotificationType.TREND_ALERT,
            title=f"Alerta de Tendencia: {trend_name}",
            message=alert_message,
            data=trend_data
        )
        
        if webhook_url:
            self.notification_handler.register_webhook(webhook_url)
        
        await self.notification_handler.send_notification(notification)


# Instancia global
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Obtener instancia global del gestor de webhooks"""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager
















