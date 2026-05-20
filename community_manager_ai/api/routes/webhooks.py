"""
Webhooks API Routes
===================

Endpoints para recibir webhooks de plataformas sociales.
"""

from fastapi import APIRouter, HTTPException, Request, Header, Depends
from typing import Optional
import json

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def get_webhook_service():
    """Dependency para obtener WebhookService"""
    from ...services.webhook_service import WebhookService
    return WebhookService()


@router.post("/{platform}")
async def receive_webhook(
    platform: str,
    request: Request,
    x_hub_signature: Optional[str] = Header(None, alias="X-Hub-Signature"),
    service = Depends(get_webhook_service)
):
    """Recibir webhook de una plataforma"""
    try:
        # Obtener payload
        payload = await request.json()
        
        # Manejar webhook
        success = service.handle_webhook(
            platform=platform,
            payload=payload,
            signature=x_hub_signature
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Error procesando webhook")
        
        return {"status": "ok"}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Payload inválido")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{platform}/verify")
async def verify_webhook(
    platform: str,
    hub_mode: Optional[str] = None,
    hub_verify_token: Optional[str] = None,
    hub_challenge: Optional[str] = None,
    service = Depends(get_webhook_service)
):
    """Verificar webhook (para Facebook/Instagram)"""
    try:
        # Verificar token
        expected_token = service.verification_tokens.get(platform)
        
        if hub_mode == "subscribe" and hub_verify_token == expected_token:
            return int(hub_challenge) if hub_challenge else {"status": "verified"}
        else:
            raise HTTPException(status_code=403, detail="Token de verificación inválido")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




