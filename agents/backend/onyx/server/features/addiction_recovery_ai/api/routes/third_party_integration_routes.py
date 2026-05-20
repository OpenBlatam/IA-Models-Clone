"""
Third party integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.third_party_integration_service import ThirdPartyIntegrationService
except ImportError:
    from ...services.third_party_integration_service import ThirdPartyIntegrationService

router = APIRouter()

third_party_integration = ThirdPartyIntegrationService()


@router.post("/integrations/connect")
async def connect_integration(
    user_id: str = Body(...),
    integration_type: str = Body(...),
    app_name: str = Body(...),
    credentials: Dict = Body(...)
):
    """Conecta una integración"""
    try:
        integration = third_party_integration.connect_integration(
            user_id, integration_type, app_name, credentials
        )
        return JSONResponse(content=integration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando integración: {str(e)}")


@router.get("/integrations/available")
async def get_available_integrations(integration_type: Optional[str] = Query(None)):
    """Obtiene integraciones disponibles"""
    try:
        integrations = third_party_integration.get_available_integrations(integration_type)
        return JSONResponse(content={
            "integrations": integrations,
            "total": len(integrations),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo integraciones: {str(e)}")



