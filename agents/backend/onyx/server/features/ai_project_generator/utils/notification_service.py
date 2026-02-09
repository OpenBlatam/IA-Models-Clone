"""
Notification Service - Servicio de Notificaciones
==================================================

Gestiona notificaciones a múltiples canales.
"""

import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class NotificationService:
    """Servicio de notificaciones multi-canal"""

    def __init__(self):
        """Inicializa el servicio de notificaciones"""
        self.channels = {
            "email": [],
            "slack": [],
            "discord": [],
            "telegram": [],
        }

    def register_channel(
        self,
        channel_type: str,
        config: Dict[str, Any],
    ) -> str:
        """
        Registra un canal de notificaciones.

        Args:
            channel_type: Tipo de canal (email, slack, discord, telegram)
            config: Configuración del canal

        Returns:
            ID del canal registrado
        """
        channel_id = f"{channel_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        if channel_type not in self.channels:
            raise ValueError(f"Canal {channel_type} no soportado")

        self.channels[channel_type].append({
            "id": channel_id,
            "config": config,
            "active": True,
            "created_at": datetime.now().isoformat(),
        })

        logger.info(f"Canal de notificación registrado: {channel_id}")
        return channel_id

    async def send_notification(
        self,
        message: str,
        title: Optional[str] = None,
        channels: Optional[List[str]] = None,
        priority: str = "normal",
    ):
        """
        Envía una notificación a los canales configurados.

        Args:
            message: Mensaje a enviar
            title: Título (opcional)
            channels: Canales específicos (si None, envía a todos)
            priority: Prioridad (low, normal, high, urgent)
        """
        if channels is None:
            channels = list(self.channels.keys())

        for channel_type in channels:
            if channel_type not in self.channels:
                continue

            for channel in self.channels[channel_type]:
                if not channel.get("active", True):
                    continue

                try:
                    if channel_type == "slack":
                        await self._send_slack(channel["config"], message, title)
                    elif channel_type == "discord":
                        await self._send_discord(channel["config"], message, title)
                    elif channel_type == "telegram":
                        await self._send_telegram(channel["config"], message, title)
                    elif channel_type == "email":
                        await self._send_email(channel["config"], message, title)
                except Exception as e:
                    logger.error(f"Error enviando notificación a {channel_type}: {e}")

    async def _send_slack(
        self,
        config: Dict[str, Any],
        message: str,
        title: Optional[str],
    ):
        """Envía notificación a Slack"""
        webhook_url = config.get("webhook_url")
        if not webhook_url:
            return

        payload = {
            "text": title or "AI Project Generator",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": title or "Notificación",
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message,
                    }
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=payload)

    async def _send_discord(
        self,
        config: Dict[str, Any],
        message: str,
        title: Optional[str],
    ):
        """Envía notificación a Discord"""
        webhook_url = config.get("webhook_url")
        if not webhook_url:
            return

        payload = {
            "embeds": [{
                "title": title or "AI Project Generator",
                "description": message,
                "color": 0x3498db,
                "timestamp": datetime.now().isoformat(),
            }]
        }

        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=payload)

    async def _send_telegram(
        self,
        config: Dict[str, Any],
        message: str,
        title: Optional[str],
    ):
        """Envía notificación a Telegram"""
        bot_token = config.get("bot_token")
        chat_id = config.get("chat_id")
        if not bot_token or not chat_id:
            return

        text = f"*{title or 'Notificación'}*\n\n{message}"
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload)

    async def _send_email(
        self,
        config: Dict[str, Any],
        message: str,
        title: Optional[str],
    ):
        """Envía notificación por email"""
        # Implementación básica - en producción usar servicio de email
        logger.info(f"Email notification: {title} - {message}")

    def list_channels(self) -> Dict[str, List[Dict[str, Any]]]:
        """Lista todos los canales registrados"""
        return {
            channel_type: [
                {
                    "id": ch["id"],
                    "active": ch.get("active", True),
                    "created_at": ch.get("created_at"),
                }
                for ch in channels
            ]
            for channel_type, channels in self.channels.items()
        }


