"""
Webhook API Endpoints
=====================

Endpoints para gestión de webhooks y API gateway.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional, List
import logging

from ..core.webhook_system import (
    get_webhook_system,
    WebhookStatus
)
from ..core.api_gateway import (
    get_api_gateway,
    APIStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


@router.post("/webhooks")
async def create_webhook(
    webhook_id: str,
    url: str,
    name: str,
    description: str,
    events: List[str],
    secret: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Crear webhook."""
    try:
        system = get_webhook_system()
        webhook = system.create_webhook(
            webhook_id=webhook_id,
            url=url,
            name=name,
            description=description,
            events=events,
            secret=secret,
            headers=headers
        )
        return {
            "webhook_id": webhook.webhook_id,
            "name": webhook.name,
            "url": webhook.url,
            "events": webhook.events,
            "status": webhook.status.value
        }
    except Exception as e:
        logger.error(f"Error creating webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhooks")
async def list_webhooks() -> Dict[str, Any]:
    """Listar webhooks."""
    try:
        system = get_webhook_system()
        webhooks = system.list_webhooks()
        return {
            "webhooks": [
                {
                    "webhook_id": w.webhook_id,
                    "name": w.name,
                    "url": w.url,
                    "events": w.events,
                    "status": w.status.value
                }
                for w in webhooks
            ],
            "count": len(webhooks)
        }
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhooks/{webhook_id}/trigger")
async def trigger_webhook(
    webhook_id: str,
    event: str,
    payload: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Disparar webhook."""
    try:
        system = get_webhook_system()
        delivery = await system.trigger_webhook(webhook_id, event, payload)
        
        if not delivery:
            raise HTTPException(status_code=404, detail="Webhook not found or inactive")
        
        return {
            "delivery_id": delivery.delivery_id,
            "status_code": delivery.status_code,
            "error": delivery.error,
            "retry_count": delivery.retry_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/{event}")
async def trigger_event(
    event: str,
    payload: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Disparar evento a todos los webhooks relevantes."""
    try:
        system = get_webhook_system()
        deliveries = await system.trigger_event(event, payload)
        return {
            "event": event,
            "deliveries": [
                {
                    "delivery_id": d.delivery_id,
                    "webhook_id": d.webhook_id,
                    "status_code": d.status_code,
                    "error": d.error
                }
                for d in deliveries
            ],
            "count": len(deliveries)
        }
    except Exception as e:
        logger.error(f"Error triggering event: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhooks/{webhook_id}/deliveries")
async def get_webhook_deliveries(
    webhook_id: str,
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Obtener entregas de webhook."""
    try:
        system = get_webhook_system()
        deliveries = system.get_deliveries(webhook_id=webhook_id, limit=limit)
        return {
            "webhook_id": webhook_id,
            "deliveries": [
                {
                    "delivery_id": d.delivery_id,
                    "event": d.event,
                    "status_code": d.status_code,
                    "error": d.error,
                    "timestamp": d.timestamp
                }
                for d in deliveries
            ],
            "count": len(deliveries)
        }
    except Exception as e:
        logger.error(f"Error getting deliveries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/endpoints")
async def register_endpoint(
    endpoint_id: str,
    name: str,
    base_url: str,
    path: str,
    method: str,
    description: str,
    headers: Optional[Dict[str, str]] = None,
    auth_type: Optional[str] = None,
    auth_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Registrar endpoint de API."""
    try:
        gateway = get_api_gateway()
        endpoint = gateway.register_endpoint(
            endpoint_id=endpoint_id,
            name=name,
            base_url=base_url,
            path=path,
            method=method,
            description=description,
            headers=headers,
            auth_type=auth_type,
            auth_config=auth_config
        )
        return {
            "endpoint_id": endpoint.endpoint_id,
            "name": endpoint.name,
            "base_url": endpoint.base_url,
            "path": endpoint.path,
            "method": endpoint.method
        }
    except Exception as e:
        logger.error(f"Error registering endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/endpoints")
async def list_endpoints(
    status: Optional[str] = None
) -> Dict[str, Any]:
    """Listar endpoints de API."""
    try:
        gateway = get_api_gateway()
        api_status = APIStatus(status.lower()) if status else None
        endpoints = gateway.list_endpoints(status=api_status)
        return {
            "endpoints": [
                {
                    "endpoint_id": e.endpoint_id,
                    "name": e.name,
                    "base_url": e.base_url,
                    "path": e.path,
                    "method": e.method,
                    "status": e.status.value
                }
                for e in endpoints
            ],
            "count": len(endpoints)
        }
    except Exception as e:
        logger.error(f"Error listing endpoints: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






