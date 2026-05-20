"""
Webhooks - Sistema de webhooks
===============================

Sistema de webhooks para notificaciones y eventos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

import aiohttp

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Tipos de eventos de webhook."""
    SESSION_CREATED = "session_created"
    SESSION_PAUSED = "session_paused"
    SESSION_RESUMED = "session_resumed"
    SESSION_STOPPED = "session_stopped"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class Webhook:
    """Configuración de webhook."""
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    enabled: bool = True
    timeout: float = 5.0
    retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebhookManager:
    """Gestor de webhooks."""
    
    def __init__(self):
        self.webhooks: List[Webhook] = []
        self._lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Obtener sesión HTTP (lazy initialization)."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def register(self, webhook: Webhook):
        """Registrar un webhook."""
        async with self._lock:
            self.webhooks.append(webhook)
            logger.info(f"Registered webhook: {webhook.url} for events: {[e.value for e in webhook.events]}")
    
    async def unregister(self, url: str):
        """Desregistrar un webhook."""
        async with self._lock:
            self.webhooks = [w for w in self.webhooks if w.url != url]
            logger.info(f"Unregistered webhook: {url}")
    
    async def trigger(self, event: WebhookEvent, data: Dict[str, Any]):
        """
        Disparar un evento a todos los webhooks suscritos.
        
        Args:
            event: Tipo de evento
            data: Datos del evento
        """
        async with self._lock:
            webhooks_to_trigger = [
                w for w in self.webhooks
                if w.enabled and event in w.events
            ]
        
        if not webhooks_to_trigger:
            return
        
        # Disparar en paralelo
        tasks = [
            self._send_webhook(webhook, event, data)
            for webhook in webhooks_to_trigger
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        data: Dict[str, Any],
    ):
        """Enviar webhook."""
        payload = {
            "event": event.value,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        
        # Agregar firma si hay secret
        headers = {"Content-Type": "application/json"}
        if webhook.secret:
            # Simplificado - en producción usar HMAC
            headers["X-Webhook-Signature"] = webhook.secret
        
        session = await self._get_session()
        
        for attempt in range(webhook.retries):
            try:
                async with session.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=webhook.timeout),
                ) as response:
                    if response.status < 400:
                        logger.debug(f"Webhook sent successfully: {webhook.url}")
                        return
                    else:
                        logger.warning(
                            f"Webhook returned {response.status}: {webhook.url}"
                        )
            
            except asyncio.TimeoutError:
                logger.warning(f"Webhook timeout (attempt {attempt + 1}/{webhook.retries}): {webhook.url}")
            except Exception as e:
                logger.error(f"Error sending webhook (attempt {attempt + 1}/{webhook.retries}): {e}")
            
            if attempt < webhook.retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def cleanup(self):
        """Limpiar recursos."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def get_webhooks(self) -> List[Webhook]:
        """Obtener todos los webhooks registrados."""
        return self.webhooks.copy()
































