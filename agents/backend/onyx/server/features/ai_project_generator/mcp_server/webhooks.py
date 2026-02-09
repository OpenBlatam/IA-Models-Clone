"""
MCP Webhooks - Sistema de notificaciones webhook
=================================================
"""

import asyncio
import httpx
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Tipos de eventos webhook"""
    RESOURCE_CREATED = "resource.created"
    RESOURCE_UPDATED = "resource.updated"
    RESOURCE_DELETED = "resource.deleted"
    QUERY_EXECUTED = "query.executed"
    ERROR_OCCURRED = "error.occurred"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"


class Webhook(BaseModel):
    """Configuración de webhook"""
    url: str = Field(..., description="URL del webhook")
    events: List[WebhookEvent] = Field(..., description="Eventos a suscribir")
    secret: Optional[str] = Field(None, description="Secret para validación")
    timeout: float = Field(default=5.0, description="Timeout en segundos")
    retry_count: int = Field(default=3, description="Número de reintentos")
    enabled: bool = Field(default=True, description="Webhook habilitado")


class WebhookPayload(BaseModel):
    """Payload de webhook"""
    event: WebhookEvent = Field(..., description="Tipo de evento")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(..., description="Datos del evento")
    resource_id: Optional[str] = Field(None, description="ID del recurso")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebhookManager:
    """
    Gestor de webhooks
    
    Permite registrar webhooks y enviar notificaciones
    cuando ocurren eventos en el servidor MCP.
    """
    
    def __init__(self):
        self._webhooks: List[Webhook] = []
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Obtiene o crea cliente HTTP"""
        if not self._client:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client
    
    def register(self, webhook: Webhook):
        """
        Registra un webhook
        
        Args:
            webhook: Configuración del webhook
        """
        self._webhooks.append(webhook)
        logger.info(f"Registered webhook: {webhook.url} for events: {webhook.events}")
    
    def unregister(self, url: str):
        """
        Elimina un webhook
        
        Args:
            url: URL del webhook a eliminar
        """
        self._webhooks = [w for w in self._webhooks if w.url != url]
        logger.info(f"Unregistered webhook: {url}")
    
    def get_webhooks(self, event: Optional[WebhookEvent] = None) -> List[Webhook]:
        """
        Obtiene webhooks registrados
        
        Args:
            event: Filtrar por evento (opcional)
            
        Returns:
            Lista de webhooks
        """
        if event:
            return [w for w in self._webhooks if event in w.events and w.enabled]
        return [w for w in self._webhooks if w.enabled]
    
    async def trigger(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Dispara un evento webhook
        
        Args:
            event: Tipo de evento
            data: Datos del evento
            resource_id: ID del recurso (opcional)
            metadata: Metadata adicional (opcional)
        """
        webhooks = self.get_webhooks(event)
        
        if not webhooks:
            return
        
        payload = WebhookPayload(
            event=event,
            data=data,
            resource_id=resource_id,
            metadata=metadata or {},
        )
        
        # Enviar a todos los webhooks suscritos
        tasks = [self._send_webhook(webhook, payload) for webhook in webhooks]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(self, webhook: Webhook, payload: WebhookPayload):
        """
        Envía un webhook
        
        Args:
            webhook: Configuración del webhook
            payload: Payload a enviar
        """
        client = await self._get_client()
        
        # Preparar headers
        headers = {
            "Content-Type": "application/json",
            "X-MCP-Event": payload.event.value,
            "X-MCP-Timestamp": payload.timestamp.isoformat(),
        }
        
        if webhook.secret:
            import hmac
            import hashlib
            import json
            payload_str = json.dumps(payload.dict(), sort_keys=True)
            signature = hmac.new(
                webhook.secret.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-MCP-Signature"] = signature
        
        # Reintentos
        last_error = None
        for attempt in range(webhook.retry_count):
            try:
                response = await client.post(
                    webhook.url,
                    json=payload.dict(),
                    headers=headers,
                    timeout=webhook.timeout,
                )
                response.raise_for_status()
                logger.info(f"Webhook sent successfully to {webhook.url}")
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Webhook attempt {attempt + 1}/{webhook.retry_count} failed for {webhook.url}: {e}"
                )
                if attempt < webhook.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to send webhook to {webhook.url} after {webhook.retry_count} attempts: {last_error}")
    
    async def close(self):
        """Cierra cliente HTTP"""
        if self._client:
            await self._client.aclose()
            self._client = None

