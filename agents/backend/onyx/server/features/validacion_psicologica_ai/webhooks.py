"""
Sistema de Webhooks para Validación Psicológica AI
==================================================
Notificaciones y webhooks para eventos del sistema
"""

from typing import List, Dict, Any, Optional, Callable
from uuid import UUID
from datetime import datetime
from enum import Enum
import structlog
import asyncio
import aiohttp
from abc import ABC, abstractmethod

from .models import ValidationStatus, PsychologicalValidation

logger = structlog.get_logger()


class WebhookEvent(str, Enum):
    """Tipos de eventos para webhooks"""
    VALIDATION_CREATED = "validation.created"
    VALIDATION_STARTED = "validation.started"
    VALIDATION_COMPLETED = "validation.completed"
    VALIDATION_FAILED = "validation.failed"
    PROFILE_GENERATED = "profile.generated"
    REPORT_GENERATED = "report.generated"
    ALERT_CREATED = "alert.created"
    CONNECTION_ESTABLISHED = "connection.established"
    CONNECTION_EXPIRED = "connection.expired"


class Webhook:
    """Representa un webhook"""
    
    def __init__(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        active: bool = True
    ):
        self.id = UUID()
        self.url = url
        self.events = events
        self.secret = secret
        self.active = active
        self.created_at = datetime.utcnow()
        self.last_triggered = None
        self.failure_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": str(self.id),
            "url": self.url,
            "events": [e.value for e in self.events],
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "failure_count": self.failure_count
        }


class WebhookPayload:
    """Payload para webhook"""
    
    def __init__(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        self.event = event
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()
        self.id = f"{event.value}_{self.timestamp.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "event": self.event.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "version": "1.0"
        }


class WebhookDelivery:
    """Entrega de webhook"""
    
    def __init__(
        self,
        webhook_id: UUID,
        payload: WebhookPayload,
        status: str,
        response_code: Optional[int] = None,
        response_body: Optional[str] = None,
        error: Optional[str] = None
    ):
        self.webhook_id = webhook_id
        self.payload = payload
        self.status = status  # "success", "failed", "pending"
        self.response_code = response_code
        self.response_body = response_body
        self.error = error
        self.delivered_at = datetime.utcnow()
        self.attempts = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "webhook_id": str(self.webhook_id),
            "payload_id": self.payload.id,
            "status": self.status,
            "response_code": self.response_code,
            "error": self.error,
            "delivered_at": self.delivered_at.isoformat(),
            "attempts": self.attempts
        }


class WebhookManager:
    """Gestor de webhooks"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._webhooks: Dict[UUID, Webhook] = {}
        self._deliveries: List[WebhookDelivery] = []
        self._timeout = 10  # segundos
        logger.info("WebhookManager initialized")
    
    def register_webhook(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None
    ) -> Webhook:
        """
        Registrar un nuevo webhook
        
        Args:
            url: URL del webhook
            events: Lista de eventos a escuchar
            secret: Secreto para validación (opcional)
            
        Returns:
            Webhook registrado
        """
        webhook = Webhook(url, events, secret)
        self._webhooks[webhook.id] = webhook
        
        logger.info(
            "Webhook registered",
            webhook_id=str(webhook.id),
            url=url,
            events=[e.value for e in events]
        )
        
        return webhook
    
    def unregister_webhook(self, webhook_id: UUID) -> bool:
        """
        Desregistrar webhook
        
        Args:
            webhook_id: ID del webhook
            
        Returns:
            True si se desregistró exitosamente
        """
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            logger.info("Webhook unregistered", webhook_id=str(webhook_id))
            return True
        return False
    
    async def trigger_webhook(
        self,
        event: WebhookEvent,
        data: Dict[str, Any]
    ) -> List[WebhookDelivery]:
        """
        Disparar webhooks para un evento
        
        Args:
            event: Evento a disparar
            data: Datos del evento
            
        Returns:
            Lista de entregas
        """
        payload = WebhookPayload(event, data)
        deliveries = []
        
        # Encontrar webhooks que escuchan este evento
        relevant_webhooks = [
            w for w in self._webhooks.values()
            if w.active and event in w.events
        ]
        
        if not relevant_webhooks:
            logger.debug("No webhooks registered for event", event=event.value)
            return deliveries
        
        # Disparar cada webhook
        tasks = [
            self._deliver_webhook(webhook, payload)
            for webhook in relevant_webhooks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, WebhookDelivery):
                deliveries.append(result)
                self._deliveries.append(result)
            elif isinstance(result, Exception):
                logger.error("Error delivering webhook", error=str(result))
        
        logger.info(
            "Webhooks triggered",
            event=event.value,
            webhooks_count=len(relevant_webhooks),
            successful=len([d for d in deliveries if d.status == "success"])
        )
        
        return deliveries
    
    async def _deliver_webhook(
        self,
        webhook: Webhook,
        payload: WebhookPayload
    ) -> WebhookDelivery:
        """
        Entregar webhook individual
        
        Args:
            webhook: Webhook a entregar
            payload: Payload a enviar
            
        Returns:
            Entrega de webhook
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PsychologicalValidationAI/1.2.0"
        }
        
        if webhook.secret:
            # En producción, usar HMAC para firmar
            headers["X-Webhook-Secret"] = webhook.secret
        
        payload_dict = payload.to_dict()
        
        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    webhook.url,
                    json=payload_dict,
                    headers=headers
                ) as response:
                    status = "success" if 200 <= response.status < 300 else "failed"
                    response_body = await response.text()
                    
                    delivery = WebhookDelivery(
                        webhook_id=webhook.id,
                        payload=payload,
                        status=status,
                        response_code=response.status,
                        response_body=response_body[:500]  # Limitar tamaño
                    )
                    
                    if status == "success":
                        webhook.failure_count = 0
                    else:
                        webhook.failure_count += 1
                        if webhook.failure_count >= 5:
                            webhook.active = False
                            logger.warning(
                                "Webhook deactivated due to failures",
                                webhook_id=str(webhook.id)
                            )
                    
                    webhook.last_triggered = datetime.utcnow()
                    
                    return delivery
                    
        except asyncio.TimeoutError:
            webhook.failure_count += 1
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                payload=payload,
                status="failed",
                error="Timeout"
            )
            return delivery
            
        except Exception as e:
            webhook.failure_count += 1
            logger.error(
                "Error delivering webhook",
                error=str(e),
                webhook_id=str(webhook.id)
            )
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                payload=payload,
                status="failed",
                error=str(e)
            )
            return delivery
    
    def get_webhooks(self, active_only: bool = False) -> List[Webhook]:
        """
        Obtener webhooks registrados
        
        Args:
            active_only: Solo webhooks activos
            
        Returns:
            Lista de webhooks
        """
        webhooks = list(self._webhooks.values())
        if active_only:
            webhooks = [w for w in webhooks if w.active]
        return webhooks
    
    def get_deliveries(
        self,
        webhook_id: Optional[UUID] = None,
        limit: int = 100
    ) -> List[WebhookDelivery]:
        """
        Obtener entregas de webhooks
        
        Args:
            webhook_id: Filtrar por webhook
            limit: Límite de resultados
            
        Returns:
            Lista de entregas
        """
        deliveries = self._deliveries
        
        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]
        
        deliveries.sort(key=lambda x: x.delivered_at, reverse=True)
        return deliveries[:limit]


# Instancia global del gestor de webhooks
webhook_manager = WebhookManager()




