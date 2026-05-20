"""
Webhooks endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.webhooks import WebhooksService, WebhookEvent

router = APIRouter()
webhooks_service = WebhooksService()


@router.post("/create/{user_id}")
async def create_webhook(
    user_id: str,
    url: str,
    events: List[str],
    secret: Optional[str] = None
) -> Dict[str, Any]:
    """Crear webhook"""
    try:
        event_enums = [WebhookEvent(e) for e in events]
        webhook = webhooks_service.create_webhook(user_id, url, event_enums, secret)
        
        return {
            "id": webhook.id,
            "url": webhook.url,
            "events": [e.value for e in webhook.events],
            "active": webhook.active,
            "secret": webhook.secret,  # En producción, no retornar secret
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deliveries/{webhook_id}")
async def get_webhook_deliveries(
    webhook_id: str,
    limit: int = 50
) -> Dict[str, Any]:
    """Obtener entregas de webhook"""
    try:
        deliveries = webhooks_service.get_webhook_deliveries(webhook_id, limit)
        return {
            "webhook_id": webhook_id,
            "deliveries": deliveries,
            "total": len(deliveries),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

