"""
Webhooks Service - Sistema de webhooks
======================================

Sistema para gestionar webhooks y eventos externos.
"""

import logging
import httpx
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Eventos de webhook"""
    APPLICATION_SUBMITTED = "application_submitted"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    OFFER_RECEIVED = "offer_received"
    SKILL_LEARNED = "skill_learned"
    LEVEL_UP = "level_up"
    BADGE_EARNED = "badge_earned"
    JOB_MATCHED = "job_matched"


@dataclass
class Webhook:
    """Webhook"""
    id: str
    user_id: str
    url: str
    events: List[WebhookEvent]
    secret: str
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


@dataclass
class WebhookDelivery:
    """Entrega de webhook"""
    id: str
    webhook_id: str
    event: WebhookEvent
    payload: Dict[str, Any]
    status: str  # success, failed, pending
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    attempted_at: datetime = field(default_factory=datetime.now)
    delivered_at: Optional[datetime] = None


class WebhooksService:
    """Servicio de webhooks"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.webhooks: Dict[str, List[Webhook]] = {}  # user_id -> webhooks
        self.deliveries: Dict[str, List[WebhookDelivery]] = {}  # webhook_id -> deliveries
        logger.info("WebhooksService initialized")
    
    def create_webhook(
        self,
        user_id: str,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None
    ) -> Webhook:
        """Crear webhook"""
        webhook_id = f"webhook_{user_id}_{int(datetime.now().timestamp())}"
        
        if not secret:
            import secrets
            secret = secrets.token_urlsafe(32)
        
        webhook = Webhook(
            id=webhook_id,
            user_id=user_id,
            url=url,
            events=events,
            secret=secret,
        )
        
        if user_id not in self.webhooks:
            self.webhooks[user_id] = []
        
        self.webhooks[user_id].append(webhook)
        
        logger.info(f"Webhook created: {webhook_id}")
        return webhook
    
    async def trigger_webhook(
        self,
        user_id: str,
        event: WebhookEvent,
        payload: Dict[str, Any]
    ) -> List[WebhookDelivery]:
        """Disparar webhook para un evento"""
        user_webhooks = self.webhooks.get(user_id, [])
        active_webhooks = [
            w for w in user_webhooks
            if w.active and event in w.events
        ]
        
        deliveries = []
        
        for webhook in active_webhooks:
            delivery = await self._deliver_webhook(webhook, event, payload)
            deliveries.append(delivery)
            
            # Actualizar estadísticas
            if delivery.status == "success":
                webhook.success_count += 1
            else:
                webhook.failure_count += 1
            
            webhook.last_triggered = datetime.now()
        
        return deliveries
    
    async def _deliver_webhook(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        payload: Dict[str, Any]
    ) -> WebhookDelivery:
        """Entregar webhook"""
        delivery_id = f"delivery_{webhook.id}_{int(datetime.now().timestamp())}"
        
        # Preparar payload con firma
        signed_payload = self._sign_payload(payload, webhook.secret)
        
        delivery = WebhookDelivery(
            id=delivery_id,
            webhook_id=webhook.id,
            event=event,
            payload=signed_payload,
            status="pending",
        )
        
        try:
            # Enviar webhook
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook.url,
                    json=signed_payload,
                    headers={
                        "X-Webhook-Event": event.value,
                        "X-Webhook-Signature": self._generate_signature(signed_payload, webhook.secret),
                    },
                    timeout=10.0,
                )
                
                delivery.response_code = response.status_code
                delivery.response_body = response.text[:500]  # Limitar tamaño
                delivery.status = "success" if response.status_code < 400 else "failed"
                delivery.delivered_at = datetime.now()
        
        except Exception as e:
            delivery.status = "failed"
            delivery.response_body = str(e)[:500]
            logger.error(f"Webhook delivery failed: {e}")
        
        if webhook.id not in self.deliveries:
            self.deliveries[webhook.id] = []
        
        self.deliveries[webhook.id].append(delivery)
        
        return delivery
    
    def _sign_payload(self, payload: Dict[str, Any], secret: str) -> Dict[str, Any]:
        """Firmar payload"""
        import hashlib
        import json
        
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hashlib.sha256(f"{payload_str}{secret}".encode()).hexdigest()
        
        return {
            **payload,
            "signature": signature,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generar firma para headers"""
        import hashlib
        import json
        
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(f"{payload_str}{secret}".encode()).hexdigest()
    
    def get_webhook_deliveries(
        self,
        webhook_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Obtener entregas de webhook"""
        deliveries = self.deliveries.get(webhook_id, [])
        
        return [
            {
                "id": d.id,
                "event": d.event.value,
                "status": d.status,
                "response_code": d.response_code,
                "attempted_at": d.attempted_at.isoformat(),
                "delivered_at": d.delivered_at.isoformat() if d.delivered_at else None,
            }
            for d in sorted(deliveries, key=lambda x: x.attempted_at, reverse=True)[:limit]
        ]
    
    def deactivate_webhook(self, webhook_id: str) -> bool:
        """Desactivar webhook"""
        for webhooks in self.webhooks.values():
            webhook = next((w for w in webhooks if w.id == webhook_id), None)
            if webhook:
                webhook.active = False
                return True
        return False




