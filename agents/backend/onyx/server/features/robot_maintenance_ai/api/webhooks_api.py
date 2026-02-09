"""
Webhooks API for external integrations and event notifications.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import logging

from .base_router import BaseRouter
from .exceptions import NotFoundError, ValidationError
from ..utils.file_helpers import get_iso_timestamp, get_timestamp_id, create_resource
from ..utils.data_helpers import ensure_resource_exists, remove_sensitive_fields

logger = logging.getLogger(__name__)

# Create base router instance
base = BaseRouter(
    prefix="/api/webhooks",
    tags=["Webhooks"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class WebhookRequest(BaseModel):
    """Request to create a webhook."""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Secret for signing webhooks")
    enabled: bool = Field(True, description="Whether webhook is enabled")


class WebhookEvent(BaseModel):
    """Webhook event data."""
    event_type: str = Field(..., description="Type of event")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: str = Field(default_factory=get_iso_timestamp)


# In-memory storage (would be database in production)
webhooks_store: Dict[str, Dict[str, Any]] = {}


@router.post("/create")
@base.timed_endpoint("create_webhook")
async def create_webhook(
    request: WebhookRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Create a new webhook subscription.
    """
    base.log_request("create_webhook", url=request.url, events_count=len(request.events))
    
    webhook = create_resource(
        {
            "url": request.url,
            "events": request.events,
            "enabled": request.enabled,
            "last_triggered": None,
            "trigger_count": 0,
            "failed_count": 0
        },
        id_prefix="wh_"
    )
    # Calculate secret based on webhook ID
    webhook["secret"] = request.secret or hashlib.sha256(webhook["id"].encode()).hexdigest()
    
    webhooks_store[webhook["id"]] = webhook
    
    return base.success({
        "webhook_id": webhook["id"],
        "url": request.url,
        "events": request.events,
        "secret": webhook["secret"]
    }, message="Webhook created successfully")


@router.get("/list")
@base.timed_endpoint("list_webhooks")
async def list_webhooks(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    List all webhooks.
    """
    base.log_request("list_webhooks")
    
    webhooks = []
    for webhook_id, webhook in webhooks_store.items():
        webhooks.append({
            "id": webhook_id,
            "url": webhook["url"],
            "events": webhook["events"],
            "enabled": webhook["enabled"],
            "created_at": webhook["created_at"],
            "trigger_count": webhook["trigger_count"],
            "failed_count": webhook["failed_count"]
        })
    
    return base.success({
        "webhooks": webhooks,
        "total": len(webhooks)
    })


@router.get("/{webhook_id}")
@base.timed_endpoint("get_webhook")
async def get_webhook(
    webhook_id: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get webhook details.
    """
    base.log_request("get_webhook", webhook_id=webhook_id)
    
    ensure_resource_exists(webhook_id, webhooks_store, "Webhook")
    webhook = webhooks_store[webhook_id]
    # Don't expose secret in response
    webhook = remove_sensitive_fields(webhook, ["secret"])
    
    return base.success(webhook)


@router.delete("/{webhook_id}")
@base.timed_endpoint("delete_webhook")
async def delete_webhook(
    webhook_id: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Delete a webhook.
    """
    base.log_request("delete_webhook", webhook_id=webhook_id)
    
    ensure_resource_exists(webhook_id, webhooks_store, "Webhook")
    del webhooks_store[webhook_id]
    
    return base.success(None, message=f"Webhook {webhook_id} deleted successfully")


@router.post("/{webhook_id}/test")
@base.timed_endpoint("test_webhook")
async def test_webhook(
    webhook_id: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Test a webhook by sending a test event.
    """
    base.log_request("test_webhook", webhook_id=webhook_id)
    
    ensure_resource_exists(webhook_id, webhooks_store, "Webhook")
    webhook = webhooks_store[webhook_id]
    
    if not webhook["enabled"]:
        raise ValidationError("Webhook is disabled")
    
    # In a real implementation, this would make an HTTP POST to the webhook URL
    # For now, we'll just simulate it
    test_event = {
        "event_type": "test",
        "data": {
            "message": "This is a test webhook event",
            "timestamp": get_iso_timestamp()
        },
        "timestamp": get_iso_timestamp()
    }
    
    # Simulate webhook trigger
    webhook["last_triggered"] = get_iso_timestamp()
    webhook["trigger_count"] += 1
    
    return base.success({
        "webhook_id": webhook_id,
        "test_event": test_event,
        "message": "Test event sent (simulated)"
    })


@router.get("/events/list")
@base.timed_endpoint("list_available_events")
async def list_available_events(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    List available webhook events.
    """
    base.log_request("list_available_events")
    
    events = [
        {
            "event": "maintenance.created",
            "description": "Triggered when a maintenance record is created"
        },
        {
            "event": "maintenance.completed",
            "description": "Triggered when a maintenance is completed"
        },
        {
            "event": "alert.created",
            "description": "Triggered when an alert is created"
        },
        {
            "event": "incident.created",
            "description": "Triggered when an incident is created"
        },
        {
            "event": "incident.resolved",
            "description": "Triggered when an incident is resolved"
        },
        {
            "event": "prediction.generated",
            "description": "Triggered when a maintenance prediction is generated"
        },
        {
            "event": "conversation.created",
            "description": "Triggered when a new conversation is created"
        },
        {
            "event": "model.trained",
            "description": "Triggered when a model is trained or retrained"
        }
    ]
    
    return base.success({
        "events": events,
        "total": len(events)
    })


def trigger_webhook(event_type: str, data: Dict[str, Any]):
    """
    Internal function to trigger webhooks for an event.
    This would be called from other parts of the system.
    """
    for webhook_id, webhook in webhooks_store.items():
        if not webhook["enabled"]:
            continue
        
        if event_type in webhook["events"]:
            try:
                # In a real implementation, this would make an HTTP POST
                # to webhook["url"] with the event data
                webhook["last_triggered"] = get_iso_timestamp()
                webhook["trigger_count"] += 1
                logger.info(f"Webhook {webhook_id} triggered for event {event_type}")
            except Exception as e:
                webhook["failed_count"] += 1
                logger.error(f"Error triggering webhook {webhook_id}: {e}")




