"""
Servicio de webhooks para notificaciones
"""

import json
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Tipos de eventos de webhook"""
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    COACHING_GENERATED = "coaching.generated"
    COMPARISON_COMPLETED = "comparison.completed"
    EXPORT_COMPLETED = "export.completed"


class WebhookService:
    """Servicio para gestionar webhooks"""
    
    def __init__(self):
        self.webhooks: Dict[str, Dict[str, Any]] = {}
        self.logger = logger
    
    def register_webhook(self, url: str, events: List[WebhookEvent],
                        secret: Optional[str] = None,
                        user_id: Optional[str] = None) -> str:
        """Registra un nuevo webhook"""
        webhook_id = f"wh_{datetime.now().timestamp()}_{len(self.webhooks)}"
        
        self.webhooks[webhook_id] = {
            "id": webhook_id,
            "url": url,
            "events": [e.value for e in events],
            "secret": secret,
            "user_id": user_id,
            "active": True,
            "created_at": datetime.now().isoformat(),
            "last_triggered": None,
            "success_count": 0,
            "failure_count": 0
        }
        
        self.logger.info(f"Webhook registered: {webhook_id} for {url}")
        return webhook_id
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """Elimina un webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            self.logger.info(f"Webhook unregistered: {webhook_id}")
            return True
        return False
    
    def get_webhook(self, webhook_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un webhook"""
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lista todos los webhooks"""
        webhooks = list(self.webhooks.values())
        
        if user_id:
            webhooks = [w for w in webhooks if w.get("user_id") == user_id]
        
        return webhooks
    
    async def trigger_webhook(self, event: WebhookEvent, data: Dict[str, Any]) -> None:
        """Dispara un webhook para un evento"""
        event_str = event.value
        
        # Encontrar webhooks que escuchan este evento
        matching_webhooks = [
            w for w in self.webhooks.values()
            if w.get("active") and event_str in w.get("events", [])
        ]
        
        if not matching_webhooks:
            return
        
        # Preparar payload
        payload = {
            "event": event_str,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Disparar webhooks en paralelo
        tasks = []
        for webhook in matching_webhooks:
            tasks.append(self._send_webhook(webhook, payload))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(self, webhook: Dict[str, Any], payload: Dict[str, Any]) -> None:
        """Envía un webhook individual"""
        webhook_id = webhook["id"]
        url = webhook["url"]
        secret = webhook.get("secret")
        
        # Agregar firma si hay secret
        if secret:
            import hmac
            import hashlib
            payload_str = json.dumps(payload, sort_keys=True)
            signature = hmac.new(
                secret.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            payload["signature"] = signature
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        webhook["success_count"] = webhook.get("success_count", 0) + 1
                        webhook["last_triggered"] = datetime.now().isoformat()
                        self.logger.info(f"Webhook {webhook_id} triggered successfully")
                    else:
                        webhook["failure_count"] = webhook.get("failure_count", 0) + 1
                        self.logger.warning(
                            f"Webhook {webhook_id} failed with status {response.status}"
                        )
        except Exception as e:
            webhook["failure_count"] = webhook.get("failure_count", 0) + 1
            self.logger.error(f"Error triggering webhook {webhook_id}: {e}")

