"""
Webhook Routes - Rutas para gestión de webhooks.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl

from api.utils import handle_api_errors
from core.services.webhook_service import WebhookService, WebhookEvent
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


class CreateWebhookRequest(BaseModel):
    """Request para crear webhook."""
    url: HttpUrl = Field(..., description="URL del webhook")
    events: List[str] = Field(..., min_length=1, description="Eventos a los que suscribirse")
    secret: Optional[str] = Field(None, description="Secret para firma (opcional)")
    metadata: Optional[dict] = Field(None, description="Metadata adicional")


class WebhookResponse(BaseModel):
    """Response de webhook."""
    webhook_id: str
    url: str
    events: List[str]
    enabled: bool
    created_at: str
    metadata: dict
    stats: dict


def get_webhook_service() -> WebhookService:
    """Obtener servicio de webhooks."""
    try:
        return get_service("webhook_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Webhook service no disponible")


@router.post("/", response_model=WebhookResponse)
@handle_api_errors
async def create_webhook(
    request: CreateWebhookRequest,
    webhook_service: WebhookService = Depends(get_webhook_service)
):
    """
    Crear nuevo webhook.
    
    Args:
        request: Datos del webhook
        
    Returns:
        Webhook creado
    """
    # Validar eventos
    try:
        events = [WebhookEvent(e) for e in request.events]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Evento inválido: {e}")
    
    webhook = webhook_service.register_webhook(
        url=str(request.url),
        events=events,
        secret=request.secret,
        metadata=request.metadata
    )
    
    return WebhookResponse(
        webhook_id=webhook.webhook_id,
        url=webhook.url,
        events=[e.value for e in webhook.events],
        enabled=webhook.enabled,
        created_at=webhook.created_at.isoformat(),
        metadata=webhook.metadata,
        stats=webhook.stats.copy()
    )


@router.get("/")
@handle_api_errors
async def list_webhooks(
    enabled_only: bool = False,
    webhook_service: WebhookService = Depends(get_webhook_service)
):
    """
    Listar webhooks.
    
    Args:
        enabled_only: Solo webhooks habilitados
        
    Returns:
        Lista de webhooks
    """
    webhooks = webhook_service.list_webhooks(enabled_only=enabled_only)
    return {"total": len(webhooks), "webhooks": webhooks}


@router.get("/{webhook_id}")
@handle_api_errors
async def get_webhook(
    webhook_id: str,
    webhook_service: WebhookService = Depends(get_webhook_service)
):
    """
    Obtener webhook por ID.
    
    Args:
        webhook_id: ID del webhook
        
    Returns:
        Webhook
    """
    webhook = webhook_service.get_webhook(webhook_id)
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook no encontrado")
    
    return webhook.to_dict()


@router.post("/{webhook_id}/enable")
@handle_api_errors
async def enable_webhook(
    webhook_id: str,
    webhook_service: WebhookService = Depends(get_webhook_service)
):
    """Habilitar webhook."""
    success = webhook_service.enable_webhook(webhook_id)
    if not success:
        raise HTTPException(status_code=404, detail="Webhook no encontrado")
    
    return {"message": "Webhook habilitado"}


@router.post("/{webhook_id}/disable")
@handle_api_errors
async def disable_webhook(
    webhook_id: str,
    webhook_service: WebhookService = Depends(get_webhook_service)
):
    """Deshabilitar webhook."""
    success = webhook_service.disable_webhook(webhook_id)
    if not success:
        raise HTTPException(status_code=404, detail="Webhook no encontrado")
    
    return {"message": "Webhook deshabilitado"}


@router.delete("/{webhook_id}")
@handle_api_errors
async def delete_webhook(
    webhook_id: str,
    webhook_service: WebhookService = Depends(get_webhook_service)
):
    """Eliminar webhook."""
    success = webhook_service.delete_webhook(webhook_id)
    if not success:
        raise HTTPException(status_code=404, detail="Webhook no encontrado")
    
    return {"message": "Webhook eliminado"}


@router.post("/trigger/{event}")
@handle_api_errors
async def trigger_webhook_event(
    event: str,
    data: dict,
    webhook_service: WebhookService = Depends(get_webhook_service)
):
    """
    Disparar evento de webhook manualmente (para testing).
    
    Args:
        event: Tipo de evento
        data: Datos del evento
        
    Returns:
        Resultado del envío
    """
    try:
        webhook_event = WebhookEvent(event)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Evento inválido: {event}. Válidos: {[e.value for e in WebhookEvent]}"
        )
    
    result = await webhook_service.trigger_event(webhook_event, data)
    return result


@router.get("/stats")
@handle_api_errors
async def get_webhook_stats(
    webhook_service: WebhookService = Depends(get_webhook_service)
):
    """Obtener estadísticas de webhooks."""
    return webhook_service.get_stats()

