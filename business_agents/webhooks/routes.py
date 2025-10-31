"""Webhook management API."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl
from datetime import datetime
import httpx
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

# In-memory webhook store (replace with DB in production)
webhooks: Dict[str, Dict[str, Any]] = {}


class WebhookCreate(BaseModel):
    url: HttpUrl
    events: List[str] = ["*"]  # Events to subscribe to
    secret: Optional[str] = None  # Optional webhook secret
    active: bool = True


class WebhookResponse(BaseModel):
    id: str
    url: str
    events: List[str]
    active: bool
    created_at: str


async def deliver_webhook(webhook_id: str, event: str, payload: Dict[str, Any]):
    """Deliver webhook to registered URL."""
    webhook = webhooks.get(webhook_id)
    if not webhook or not webhook["active"]:
        return
    
    # Check if webhook subscribes to this event
    if "*" not in webhook["events"] and event not in webhook["events"]:
        return
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Content-Type": "application/json"}
            if webhook.get("secret"):
                # In production, add signature header
                headers["X-Webhook-Secret"] = webhook["secret"]
            
            response = await client.post(
                str(webhook["url"]),
                json={
                    "event": event,
                    "payload": payload,
                    "timestamp": datetime.now().isoformat(),
                    "webhook_id": webhook_id
                },
                headers=headers
            )
            response.raise_for_status()
            logger.info(f"Webhook delivered: {webhook_id} -> {webhook['url']}")
            
            # Publish event to event bus
            try:
                from ..event_system import event_bus, EventType
                await event_bus.publish(EventType.WEBHOOK_TRIGGERED, {
                    "webhook_id": webhook_id,
                    "event": event,
                    "url": str(webhook["url"])
                })
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Webhook delivery failed: {webhook_id} - {e}")


@router.post("/", response_model=WebhookResponse)
async def create_webhook(webhook: WebhookCreate):
    """Register a new webhook."""
    webhook_id = str(uuid.uuid4())
    webhooks[webhook_id] = {
        "id": webhook_id,
        "url": str(webhook.url),
        "events": webhook.events,
        "secret": webhook.secret,
        "active": webhook.active,
        "created_at": datetime.now().isoformat()
    }
    return webhooks[webhook_id]


@router.get("/", response_model=List[WebhookResponse])
async def list_webhooks():
    """List all registered webhooks."""
    return list(webhooks.values())


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(webhook_id: str):
    """Get webhook details."""
    webhook = webhooks.get(webhook_id)
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return webhook


@router.delete("/{webhook_id}")
async def delete_webhook(webhook_id: str):
    """Delete a webhook."""
    if webhook_id not in webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    del webhooks[webhook_id]
    return {"status": "deleted", "webhook_id": webhook_id}


@router.post("/trigger/{event}")
async def trigger_webhook(event: str, payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """Trigger webhooks for an event (admin/testing endpoint)."""
    for webhook_id in webhooks.keys():
        background_tasks.add_task(deliver_webhook, webhook_id, event, payload)
    return {"status": "triggered", "event": event, "webhooks": len(webhooks)}

