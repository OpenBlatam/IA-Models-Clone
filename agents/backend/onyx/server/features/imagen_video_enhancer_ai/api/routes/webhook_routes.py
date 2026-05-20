"""
Webhook Routes
==============

API routes for webhook management.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..dependencies import get_agent
from ..models import WebhookRegisterRequest
from ...core.webhook_manager import Webhook, WebhookEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/register")
async def register_webhook(request: WebhookRegisterRequest):
    """Register a webhook."""
    agent = get_agent()
    
    try:
        # Convert event strings to WebhookEvent enums
        events = [WebhookEvent(e) for e in request.events]
        
        webhook = Webhook(
            url=request.url,
            events=events,
            secret=request.secret,
            timeout=request.timeout,
            retries=request.retries,
            enabled=request.enabled
        )
        
        agent.webhook_manager.register(webhook)
        
        return JSONResponse({
            "success": True,
            "message": "Webhook registered successfully"
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {e}")
    except Exception as e:
        logger.error(f"Error registering webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/unregister")
async def unregister_webhook(url: str):
    """Unregister a webhook."""
    agent = get_agent()
    
    try:
        agent.webhook_manager.unregister(url)
        return JSONResponse({
            "success": True,
            "message": "Webhook unregistered successfully"
        })
    except Exception as e:
        logger.error(f"Error unregistering webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




