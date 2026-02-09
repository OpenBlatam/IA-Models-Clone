"""
Webhooks Router - Endpoints for webhook management.

This module provides REST API endpoints for managing
webhooks.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends

from ..models import WebhookRequest, ErrorResponse
from ..auth import verify_token
from ..webhooks import WebhookManager, WebhookEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])

# Initialize manager (should be injected in production)
webhook_manager = WebhookManager()


@router.get("", response_model=dict)
async def list_webhooks(
    token: str = Depends(verify_token),
):
    """
    List webhooks.
    
    Args:
        token: Authentication token
    
    Returns:
        Dictionary with webhooks list
    """
    try:
        webhooks = list(webhook_manager.webhooks.values())
        return {"webhooks": [w.to_dict() if hasattr(w, 'to_dict') else w for w in webhooks]}
    except Exception as e:
        logger.exception("Error listing webhooks")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=dict)
async def register_webhook(
    request: WebhookRequest,
    token: str = Depends(verify_token),
):
    """
    Register a webhook.
    
    Args:
        request: Webhook registration request
        token: Authentication token
    
    Returns:
        Webhook dictionary
    """
    try:
        events = [WebhookEvent(e) for e in request.events]
        webhook = webhook_manager.register_webhook(
            url=request.url,
            events=events,
            secret=request.secret,
        )
        return webhook.to_dict() if hasattr(webhook, 'to_dict') else webhook
    except Exception as e:
        logger.exception("Error registering webhook")
        raise HTTPException(status_code=500, detail=str(e))


