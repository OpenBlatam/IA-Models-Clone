"""
API de Webhooks

Endpoints para:
- Registrar webhooks
- Listar webhooks
- Obtener estadísticas
- Eliminar webhooks
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from middleware.auth_middleware import require_role
from services.webhooks import (
    get_webhook_service,
    WebhookEvent
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/webhooks",
    tags=["webhooks"],
    dependencies=[Depends(require_role("admin"))]  # Requiere rol admin
)


@router.post("/register")
async def register_webhook(
    url: str = Body(..., description="URL del webhook"),
    events: List[str] = Body(..., description="Lista de eventos a suscribir"),
    secret: Optional[str] = Body(None, description="Secreto para verificación de firma")
) -> Dict[str, Any]:
    """
    Registra un nuevo webhook.
    """
    try:
        # Validar eventos
        event_enums = []
        for event_str in events:
            try:
                event_enums.append(WebhookEvent(event_str))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid event type: {event_str}"
                )
        
        webhook_service = get_webhook_service()
        webhook_id = webhook_service.register_webhook(
            url=url,
            events=event_enums,
            secret=secret
        )
        
        return {
            "webhook_id": webhook_id,
            "message": "Webhook registered successfully",
            "url": url,
            "events": events
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering webhook: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering webhook: {str(e)}"
        )


@router.get("/list")
async def list_webhooks() -> Dict[str, Any]:
    """
    Lista todos los webhooks registrados.
    """
    try:
        webhook_service = get_webhook_service()
        webhooks = webhook_service.list_webhooks()
        
        return {
            "webhooks": webhooks,
            "total": len(webhooks)
        }
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing webhooks: {str(e)}"
        )


@router.get("/{webhook_id}/stats")
async def get_webhook_stats(webhook_id: str) -> Dict[str, Any]:
    """
    Obtiene estadísticas de un webhook.
    """
    try:
        webhook_service = get_webhook_service()
        stats = webhook_service.get_webhook_stats(webhook_id)
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook {webhook_id} not found"
            )
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting webhook stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving webhook stats: {str(e)}"
        )


@router.delete("/{webhook_id}")
async def unregister_webhook(webhook_id: str) -> Dict[str, Any]:
    """
    Elimina un webhook.
    """
    try:
        webhook_service = get_webhook_service()
        success = webhook_service.unregister_webhook(webhook_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook {webhook_id} not found"
            )
        
        return {
            "message": f"Webhook {webhook_id} unregistered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering webhook: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unregistering webhook: {str(e)}"
        )

