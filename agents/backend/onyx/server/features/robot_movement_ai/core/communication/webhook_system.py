"""
Webhook System
==============

Sistema de webhooks para notificaciones externas.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import aiohttp
import json

logger = logging.getLogger(__name__)


class WebhookStatus(Enum):
    """Estado de webhook."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


@dataclass
class Webhook:
    """Webhook."""
    webhook_id: str
    url: str
    name: str
    description: str
    events: List[str]  # Lista de eventos a escuchar
    secret: Optional[str] = None
    status: WebhookStatus = WebhookStatus.ACTIVE
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class WebhookDelivery:
    """Entrega de webhook."""
    delivery_id: str
    webhook_id: str
    event: str
    payload: Dict[str, Any]
    status_code: Optional[int] = None
    response: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    retry_count: int = 0


class WebhookSystem:
    """
    Sistema de webhooks.
    
    Gestiona webhooks y entrega de eventos.
    """
    
    def __init__(self):
        """Inicializar sistema de webhooks."""
        self.webhooks: Dict[str, Webhook] = {}
        self.deliveries: List[WebhookDelivery] = []
        self.max_deliveries = 10000
        self.max_retries = 3
    
    def create_webhook(
        self,
        webhook_id: str,
        url: str,
        name: str,
        description: str,
        events: List[str],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Webhook:
        """
        Crear webhook.
        
        Args:
            webhook_id: ID único del webhook
            url: URL del webhook
            name: Nombre
            description: Descripción
            events: Lista de eventos
            secret: Secreto para firma (opcional)
            headers: Headers adicionales
            
        Returns:
            Webhook creado
        """
        webhook = Webhook(
            webhook_id=webhook_id,
            url=url,
            name=name,
            description=description,
            events=events,
            secret=secret,
            headers=headers or {},
            status=WebhookStatus.ACTIVE
        )
        
        self.webhooks[webhook_id] = webhook
        logger.info(f"Created webhook: {name} ({webhook_id})")
        
        return webhook
    
    async def trigger_webhook(
        self,
        webhook_id: str,
        event: str,
        payload: Dict[str, Any]
    ) -> Optional[WebhookDelivery]:
        """
        Disparar webhook.
        
        Args:
            webhook_id: ID del webhook
            event: Nombre del evento
            payload: Datos del evento
            
        Returns:
            Entrega de webhook o None
        """
        if webhook_id not in self.webhooks:
            return None
        
        webhook = self.webhooks[webhook_id]
        
        # Verificar si el evento está en la lista
        if event not in webhook.events:
            return None
        
        # Verificar si está activo
        if webhook.status != WebhookStatus.ACTIVE:
            return None
        
        # Crear payload
        webhook_payload = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": payload
        }
        
        # Firmar si hay secreto
        if webhook.secret:
            import hmac
            import hashlib
            signature = hmac.new(
                webhook.secret.encode(),
                json.dumps(webhook_payload).encode(),
                hashlib.sha256
            ).hexdigest()
            webhook_payload["signature"] = signature
        
        # Enviar webhook
        delivery = await self._send_webhook(webhook, event, webhook_payload)
        
        return delivery
    
    async def _send_webhook(
        self,
        webhook: Webhook,
        event: str,
        payload: Dict[str, Any]
    ) -> WebhookDelivery:
        """Enviar webhook."""
        delivery_id = f"delivery_{len(self.deliveries)}"
        delivery = WebhookDelivery(
            delivery_id=delivery_id,
            webhook_id=webhook.webhook_id,
            event=event,
            payload=payload
        )
        
        try:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "RobotMovementAI/1.0",
                **webhook.headers
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    delivery.status_code = response.status
                    delivery.response = await response.text()
                    
                    if response.status >= 400:
                        delivery.error = f"HTTP {response.status}"
                        if delivery.retry_count < self.max_retries:
                            # Reintentar
                            await asyncio.sleep(2 ** delivery.retry_count)
                            delivery.retry_count += 1
                            return await self._send_webhook(webhook, event, payload)
        except Exception as e:
            delivery.error = str(e)
            if delivery.retry_count < self.max_retries:
                # Reintentar
                await asyncio.sleep(2 ** delivery.retry_count)
                delivery.retry_count += 1
                return await self._send_webhook(webhook, event, payload)
        
        # Guardar entrega
        self.deliveries.append(delivery)
        if len(self.deliveries) > self.max_deliveries:
            self.deliveries = self.deliveries[-self.max_deliveries:]
        
        return delivery
    
    async def trigger_event(
        self,
        event: str,
        payload: Dict[str, Any]
    ) -> List[WebhookDelivery]:
        """
        Disparar evento a todos los webhooks relevantes.
        
        Args:
            event: Nombre del evento
            payload: Datos del evento
            
        Returns:
            Lista de entregas
        """
        deliveries = []
        
        for webhook in self.webhooks.values():
            if event in webhook.events and webhook.status == WebhookStatus.ACTIVE:
                delivery = await self.trigger_webhook(webhook.webhook_id, event, payload)
                if delivery:
                    deliveries.append(delivery)
        
        return deliveries
    
    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Obtener webhook por ID."""
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self) -> List[Webhook]:
        """Listar todos los webhooks."""
        return list(self.webhooks.values())
    
    def update_webhook_status(
        self,
        webhook_id: str,
        status: WebhookStatus
    ) -> bool:
        """Actualizar estado de webhook."""
        if webhook_id not in self.webhooks:
            return False
        
        self.webhooks[webhook_id].status = status
        self.webhooks[webhook_id].updated_at = datetime.now().isoformat()
        return True
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """Eliminar webhook."""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            return True
        return False
    
    def get_deliveries(
        self,
        webhook_id: Optional[str] = None,
        event: Optional[str] = None,
        limit: int = 100
    ) -> List[WebhookDelivery]:
        """
        Obtener entregas.
        
        Args:
            webhook_id: Filtrar por webhook
            event: Filtrar por evento
            limit: Límite de resultados
            
        Returns:
            Lista de entregas
        """
        deliveries = self.deliveries
        
        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]
        if event:
            deliveries = [d for d in deliveries if d.event == event]
        
        deliveries.sort(key=lambda x: x.timestamp, reverse=True)
        return deliveries[:limit]


# Instancia global
_webhook_system: Optional[WebhookSystem] = None


def get_webhook_system() -> WebhookSystem:
    """Obtener instancia global del sistema de webhooks."""
    global _webhook_system
    if _webhook_system is None:
        _webhook_system = WebhookSystem()
    return _webhook_system

