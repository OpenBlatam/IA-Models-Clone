"""
Webhooks - Sistema de webhooks
===============================

Sistema para enviar webhooks a URLs externas.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import httpx

from .error_handling import safe_async_call

logger = logging.getLogger(__name__)


@dataclass
class WebhookEvent:
    """Evento de webhook."""
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    webhook_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class WebhookConfig:
    """Configuración de webhook."""
    url: str
    secret: Optional[str] = None
    events: List[str] = None
    timeout: int = 30
    retries: int = 3
    enabled: bool = True
    
    def __post_init__(self):
        from .validation_utils import validate_not_empty, validate_positive, validate_non_negative
        
        if self.events is None:
            self.events = ["*"]  # Todos los eventos
        
        validate_not_empty(self.url, "url")
        validate_positive(self.timeout, "timeout")
        validate_non_negative(self.retries, "retries")


class WebhookManager:
    """Gestor de webhooks."""
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self._client = httpx.AsyncClient(timeout=30.0)
    
    def register(self, webhook_id: str, config: WebhookConfig) -> None:
        """
        Registrar webhook.
        
        Args:
            webhook_id: ID del webhook.
            config: Configuración del webhook.
        """
        from .validation_utils import validate_not_empty, validate_not_none
        validate_not_empty(webhook_id, "webhook_id")
        validate_not_none(config, "config")
        
        self.webhooks[webhook_id] = config
        logger.info(f"Webhook registered: {webhook_id} -> {config.url}")
    
    def unregister(self, webhook_id: str) -> None:
        """Desregistrar webhook."""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Webhook unregistered: {webhook_id}")
    
    async def send(self, event: WebhookEvent) -> Dict[str, Any]:
        """
        Enviar evento a todos los webhooks suscritos.
        
        Args:
            event: Evento a enviar.
        
        Returns:
            Resultados del envío.
        """
        results = {}
        
        for webhook_id, config in self.webhooks.items():
            if not config.enabled:
                continue
            
            # Verificar si el webhook está suscrito a este evento
            if "*" not in config.events and event.event_type not in config.events:
                continue
            
            async def send_webhook():
                return await self._send_to_webhook(webhook_id, config, event)
            
            result = await safe_async_call(
                send_webhook,
                operation=f"sending webhook {webhook_id}",
                default_return={"success": False, "error": "Unknown error"},
                logger_instance=logger,
                reraise=False
            )
            results[webhook_id] = result
        
        return results
    
    async def _send_to_webhook(
        self,
        webhook_id: str,
        config: WebhookConfig,
        event: WebhookEvent
    ) -> Dict[str, Any]:
        """Enviar evento a un webhook específico."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Cursor-Agent-24-7/1.0"
        }
        
        # Agregar signature si hay secret
        payload = event.to_dict()
        if config.secret:
            import hmac
            import hashlib
            signature = hmac.new(
                config.secret.encode(),
                json.dumps(payload, sort_keys=True).encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        # Reintentos
        from .validation_utils import validate_positive
        validate_positive(config.retries, "retries")
        
        last_error = None
        for attempt in range(config.retries):
            try:
                response = await self._client.post(
                    config.url,
                    json=payload,
                    headers=headers,
                    timeout=config.timeout
                )
                response.raise_for_status()
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "response": response.json() if response.content else None
                }
            except httpx.HTTPError as e:
                last_error = e
                if attempt < config.retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        raise last_error
    
    async def close(self) -> None:
        """Cerrar cliente HTTP."""
        await self._client.aclose()


# Manager global
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Obtener manager de webhooks global."""
    global _webhook_manager
    
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    
    return _webhook_manager




