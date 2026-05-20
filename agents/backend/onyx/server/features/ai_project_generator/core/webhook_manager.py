"""
Webhook Manager - Gestor de Webhooks
====================================

Gestión de webhooks:
- Webhook registration
- Event delivery
- Retry logic
- Signature verification
- Webhook testing
"""

import logging
import hmac
import hashlib
import json
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class WebhookStatus(str, Enum):
    """Estados de webhook"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    SUSPENDED = "suspended"


class Webhook:
    """Webhook"""
    
    def __init__(
        self,
        webhook_id: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        self.webhook_id = webhook_id
        self.url = url
        self.events = events
        self.secret = secret
        self.headers = headers or {}
        self.status = WebhookStatus.ACTIVE
        self.created_at = datetime.now()
        self.last_delivery: Optional[datetime] = None
        self.failure_count = 0
        self.max_failures = 5
    
    def generate_signature(self, payload: str) -> str:
        """Genera firma para webhook"""
        if not self.secret:
            return ""
        
        signature = hmac.new(
            self.secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    def verify_signature(self, payload: str, signature: str) -> bool:
        """Verifica firma de webhook"""
        expected = self.generate_signature(payload)
        return hmac.compare_digest(expected, signature)


class WebhookManager:
    """
    Gestor de webhooks.
    """
    
    def __init__(self) -> None:
        self.webhooks: Dict[str, Webhook] = {}
        self.delivery_history: List[Dict[str, Any]] = []
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def register_webhook(
        self,
        webhook_id: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Webhook:
        """Registra un webhook"""
        webhook = Webhook(webhook_id, url, events, secret, headers)
        self.webhooks[webhook_id] = webhook
        logger.info(f"Webhook registered: {webhook_id}")
        return webhook
    
    async def deliver_event(
        self,
        event: str,
        payload: Dict[str, Any],
        webhook_id: Optional[str] = None
    ) -> int:
        """Entrega evento a webhooks"""
        delivered = 0
        
        webhooks_to_notify = []
        if webhook_id:
            webhook = self.webhooks.get(webhook_id)
            if webhook and event in webhook.events:
                webhooks_to_notify.append(webhook)
        else:
            webhooks_to_notify = [
                w for w in self.webhooks.values()
                if event in w.events and w.status == WebhookStatus.ACTIVE
            ]
        
        for webhook in webhooks_to_notify:
            if await self._deliver_to_webhook(webhook, event, payload):
                delivered += 1
        
        return delivered
    
    async def _deliver_to_webhook(
        self,
        webhook: Webhook,
        event: str,
        payload: Dict[str, Any]
    ) -> bool:
        """Entrega a un webhook específico"""
        payload_str = json.dumps(payload)
        headers = webhook.headers.copy()
        headers["X-Webhook-Event"] = event
        headers["Content-Type"] = "application/json"
        
        # Agregar firma si hay secret
        if webhook.secret:
            signature = webhook.generate_signature(payload_str)
            headers["X-Webhook-Signature"] = signature
        
        try:
            response = await self.client.post(
                webhook.url,
                content=payload_str,
                headers=headers
            )
            
            success = response.status_code < 400
            webhook.last_delivery = datetime.now()
            
            if success:
                webhook.failure_count = 0
            else:
                webhook.failure_count += 1
                if webhook.failure_count >= webhook.max_failures:
                    webhook.status = WebhookStatus.SUSPENDED
            
            # Registrar en historial
            self.delivery_history.append({
                "webhook_id": webhook.webhook_id,
                "event": event,
                "status_code": response.status_code,
                "success": success,
                "timestamp": datetime.now().isoformat()
            })
            
            return success
            
        except Exception as e:
            logger.error(f"Webhook delivery failed: {e}")
            webhook.failure_count += 1
            if webhook.failure_count >= webhook.max_failures:
                webhook.status = WebhookStatus.SUSPENDED
            
            return False
    
    def get_webhook_stats(self, webhook_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas de un webhook"""
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return None
        
        deliveries = [d for d in self.delivery_history if d["webhook_id"] == webhook_id]
        successful = sum(1 for d in deliveries if d.get("success"))
        
        return {
            "webhook_id": webhook_id,
            "status": webhook.status.value,
            "total_deliveries": len(deliveries),
            "successful_deliveries": successful,
            "failure_count": webhook.failure_count,
            "last_delivery": webhook.last_delivery.isoformat() if webhook.last_delivery else None
        }
    
    async def test_webhook(self, webhook_id: str) -> bool:
        """Prueba un webhook"""
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return False
        
        test_payload = {
            "event": "test",
            "message": "This is a test webhook",
            "timestamp": datetime.now().isoformat()
        }
        
        return await self._deliver_to_webhook(webhook, "test", test_payload)
    
    async def close(self) -> None:
        """Cierra cliente"""
        await self.client.aclose()


def get_webhook_manager() -> WebhookManager:
    """Obtiene gestor de webhooks"""
    return WebhookManager()















