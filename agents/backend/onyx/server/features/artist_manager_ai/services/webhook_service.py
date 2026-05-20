"""
Webhook Service
===============

Servicio de webhooks para notificaciones externas.
"""

import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Eventos de webhook."""
    EVENT_CREATED = "event.created"
    EVENT_UPDATED = "event.updated"
    EVENT_DELETED = "event.deleted"
    ROUTINE_COMPLETED = "routine.completed"
    PROTOCOL_VIOLATION = "protocol.violation"
    WARDROBE_RECOMMENDATION = "wardrobe.recommendation"
    DAILY_SUMMARY = "daily_summary.generated"


@dataclass
class Webhook:
    """Webhook."""
    id: str
    artist_id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "artist_id": self.artist_id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "active": self.active,
            "created_at": self.created_at.isoformat()
        }


class WebhookService:
    """Servicio de webhooks."""
    
    def __init__(self):
        """Inicializar servicio de webhooks."""
        self.webhooks: Dict[str, Webhook] = {}
        self._logger = logger
    
    def register_webhook(
        self,
        artist_id: str,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None
    ) -> Webhook:
        """Registrar webhook."""
        import uuid
        
        webhook = Webhook(
            id=str(uuid.uuid4()),
            artist_id=artist_id,
            url=url,
            events=events,
            secret=secret
        )
        
        self.webhooks[webhook.id] = webhook
        self._logger.info(f"Registered webhook {webhook.id} for artist {artist_id}")
        return webhook
    
    async def trigger_webhook(
        self,
        artist_id: str,
        event: WebhookEvent,
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Disparar webhooks para un evento."""
        results = []
        relevant_webhooks = [
            w for w in self.webhooks.values()
            if w.artist_id == artist_id and w.active and event in w.events
        ]
        
        for webhook in relevant_webhooks:
            try:
                payload = {
                    "event": event.value,
                    "artist_id": artist_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": data
                }
                
                if webhook.secret:
                    import hmac, hashlib
                    signature = hmac.new(
                        webhook.secret.encode(),
                        str(payload).encode(),
                        hashlib.sha256
                    ).hexdigest()
                    payload["signature"] = signature
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(webhook.url, json=payload)
                    response.raise_for_status()
                    results.append({"webhook_id": webhook.id, "status": "success"})
            except Exception as e:
                self._logger.error(f"Error triggering webhook {webhook.id}: {str(e)}")
                results.append({"webhook_id": webhook.id, "status": "error", "error": str(e)})
        
        return results
