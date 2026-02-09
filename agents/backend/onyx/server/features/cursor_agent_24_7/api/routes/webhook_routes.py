"""
Webhook Routes - Rutas para webhooks
====================================

Endpoints para gestionar webhooks.
"""

import logging
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, HttpUrl

from ...core.webhooks import get_webhook_manager, WebhookConfig
from ...core.oauth2 import get_current_active_user
from ..utils import handle_route_errors, create_success_response
from ..serializers import serialize_webhooks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])


class WebhookCreateRequest(BaseModel):
    """Request para crear webhook."""
    url: HttpUrl
    secret: Optional[str] = None
    events: List[str] = ["*"]
    timeout: int = 30
    retries: int = 3


class WebhookResponse(BaseModel):
    """Response de webhook."""
    webhook_id: str
    url: str
    events: List[str]
    enabled: bool


@router.post("", response_model=WebhookResponse)
@handle_route_errors("creating webhook")
async def create_webhook(
    request: WebhookCreateRequest,
    current_user = Depends(get_current_active_user)
):
    """
    Crear un nuevo webhook.
    
    Args:
        request: Configuración del webhook.
        current_user: Usuario autenticado.
    
    Returns:
        Información del webhook creado.
    """
    webhook_id = str(uuid.uuid4())
    manager = get_webhook_manager()
    
    config = WebhookConfig(
        url=str(request.url),
        secret=request.secret,
        events=request.events,
        timeout=request.timeout,
        retries=request.retries
    )
    
    manager.register(webhook_id, config)
    
    return WebhookResponse(
        webhook_id=webhook_id,
        url=str(request.url),
        events=request.events,
        enabled=True
    )


@router.get("", response_model=List[WebhookResponse])
@handle_route_errors("listing webhooks")
async def list_webhooks(
    current_user = Depends(get_current_active_user)
):
    """Listar todos los webhooks."""
    manager = get_webhook_manager()
    
    webhooks = serialize_webhooks(manager.webhooks)
    return [WebhookResponse(**wh) for wh in webhooks]


@router.delete("/{webhook_id}")
@handle_route_errors("deleting webhook")
async def delete_webhook(
    webhook_id: str,
    current_user = Depends(get_current_active_user)
):
    """Eliminar webhook."""
    manager = get_webhook_manager()
    
    if webhook_id not in manager.webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    manager.unregister(webhook_id)
    
    return create_success_response(
        "success",
        "Webhook deleted"
    )




