"""
Webhooks routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional, List

try:
    from services.webhook_service import WebhookService
except ImportError:
    from ...services.webhook_service import WebhookService

router = APIRouter()

webhooks = WebhookService()


@router.post("/webhooks/register")
async def register_webhook(
    user_id: str = Body(...),
    url: str = Body(...),
    event_types: List[str] = Body(...),
    secret: Optional[str] = Body(None)
):
    """Registra un webhook"""
    try:
        webhook = webhooks.register_webhook(user_id, url, event_types, secret)
        return JSONResponse(content=webhook)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando webhook: {str(e)}")


@router.get("/webhooks/{user_id}")
async def get_user_webhooks(user_id: str, active_only: bool = Query(True)):
    """Obtiene webhooks del usuario"""
    try:
        webhooks_list = webhooks.get_webhooks(user_id, active_only)
        return JSONResponse(content={
            "user_id": user_id,
            "webhooks": webhooks_list,
            "total": len(webhooks_list),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo webhooks: {str(e)}")



