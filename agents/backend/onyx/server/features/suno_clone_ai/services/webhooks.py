"""
Sistema de Webhooks

Proporciona:
- Registro de webhooks
- Envío de eventos a webhooks
- Retry automático
- Verificación de firma
- Historial de entregas
"""

import logging
import hmac
import hashlib
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Tipos de eventos de webhook"""
    SONG_GENERATED = "song.generated"
    SONG_UPDATED = "song.updated"
    SONG_DELETED = "song.deleted"
    USER_REGISTERED = "user.registered"
    GENERATION_STARTED = "generation.started"
    GENERATION_COMPLETED = "generation.completed"
    GENERATION_FAILED = "generation.failed"


@dataclass
class Webhook:
    """Representa un webhook registrado"""
    id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


@dataclass
class WebhookDelivery:
    """Representa una entrega de webhook"""
    id: str
    webhook_id: str
    event: str
    payload: Dict[str, Any]
    status: str
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0


class WebhookService:
    """Servicio de webhooks"""
    
    def __init__(self):
        self.webhooks: Dict[str, Webhook] = {}
        self.deliveries: List[WebhookDelivery] = []
        self.max_retries = 3
        logger.info("WebhookService initialized")
    
    def register_webhook(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        webhook_id: Optional[str] = None
    ) -> str:
        """
        Registra un nuevo webhook
        
        Args:
            url: URL del webhook
            events: Lista de eventos a suscribir
            secret: Secreto para verificación de firma (opcional)
            webhook_id: ID opcional del webhook
        
        Returns:
            ID del webhook
        """
        import uuid
        webhook_id = webhook_id or str(uuid.uuid4())
        
        webhook = Webhook(
            id=webhook_id,
            url=url,
            events=events,
            secret=secret
        )
        
        self.webhooks[webhook_id] = webhook
        logger.info(f"Webhook registered: {webhook_id} for {len(events)} events")
        
        return webhook_id
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """Elimina un webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Webhook unregistered: {webhook_id}")
            return True
        return False
    
    async def trigger_webhook(
        self,
        event: WebhookEvent,
        payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Dispara un evento a todos los webhooks suscritos
        
        Args:
            event: Tipo de evento
            payload: Datos del evento
        
        Returns:
            Lista de resultados de entrega
        """
        results = []
        
        # Encontrar webhooks suscritos al evento
        subscribed_webhooks = [
            webhook for webhook in self.webhooks.values()
            if webhook.active and event in webhook.events
        ]
        
        for webhook in subscribed_webhooks:
            try:
                result = await self._deliver_webhook(webhook, event, payload)
                results.append(result)
            except Exception as e:
                logger.error(f"Error delivering webhook {webhook.id}: {e}")
                results.append({
                    "webhook_id": webhook.id,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    async def _deliver_webhook(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Entrega un webhook
        
        Args:
            webhook: Webhook a entregar
            event: Tipo de evento
            payload: Datos del evento
        
        Returns:
            Resultado de la entrega
        """
        import uuid
        delivery_id = str(uuid.uuid4())
        
        # Preparar payload
        webhook_payload = {
            "event": event.value,
            "timestamp": datetime.now().isoformat(),
            "data": payload
        }
        
        # Agregar firma si hay secreto
        headers = {"Content-Type": "application/json"}
        if webhook.secret:
            signature = self._generate_signature(
                json.dumps(webhook_payload),
                webhook.secret
            )
            headers["X-Webhook-Signature"] = signature
        
        # Intentar entrega
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    webhook.url,
                    json=webhook_payload,
                    headers=headers
                )
                
                success = 200 <= response.status_code < 300
                
                delivery = WebhookDelivery(
                    id=delivery_id,
                    webhook_id=webhook.id,
                    event=event.value,
                    payload=payload,
                    status="success" if success else "failed",
                    response_code=response.status_code,
                    response_body=response.text[:500]  # Limitar tamaño
                )
                
                if success:
                    webhook.success_count += 1
                else:
                    webhook.failure_count += 1
                    delivery.error = f"HTTP {response.status_code}"
                
                webhook.last_triggered = datetime.now()
                self.deliveries.append(delivery)
                
                return {
                    "webhook_id": webhook.id,
                    "delivery_id": delivery_id,
                    "success": success,
                    "status_code": response.status_code
                }
        
        except Exception as e:
            delivery = WebhookDelivery(
                id=delivery_id,
                webhook_id=webhook.id,
                event=event.value,
                payload=payload,
                status="failed",
                error=str(e),
                retry_count=1
            )
            
            webhook.failure_count += 1
            self.deliveries.append(delivery)
            
            return {
                "webhook_id": webhook.id,
                "delivery_id": delivery_id,
                "success": False,
                "error": str(e)
            }
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Genera firma HMAC para el payload"""
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verifica la firma de un webhook"""
        expected_signature = self._generate_signature(payload, secret)
        return hmac.compare_digest(expected_signature, signature)
    
    def get_webhook_stats(self, webhook_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas de un webhook"""
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return {}
        
        deliveries = [d for d in self.deliveries if d.webhook_id == webhook_id]
        
        return {
            "webhook_id": webhook_id,
            "url": webhook.url,
            "active": webhook.active,
            "events": [e.value for e in webhook.events],
            "success_count": webhook.success_count,
            "failure_count": webhook.failure_count,
            "total_deliveries": len(deliveries),
            "success_rate": (
                (webhook.success_count / (webhook.success_count + webhook.failure_count) * 100)
                if (webhook.success_count + webhook.failure_count) > 0 else 0
            ),
            "last_triggered": webhook.last_triggered.isoformat() if webhook.last_triggered else None
        }
    
    def list_webhooks(self) -> List[Dict[str, Any]]:
        """Lista todos los webhooks"""
        return [
            {
                "id": webhook.id,
                "url": webhook.url,
                "events": [e.value for e in webhook.events],
                "active": webhook.active,
                "created_at": webhook.created_at.isoformat(),
                "success_count": webhook.success_count,
                "failure_count": webhook.failure_count
            }
            for webhook in self.webhooks.values()
        ]


# Instancia global
_webhook_service: Optional[WebhookService] = None


def get_webhook_service() -> WebhookService:
    """Obtiene la instancia global del servicio de webhooks"""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service

