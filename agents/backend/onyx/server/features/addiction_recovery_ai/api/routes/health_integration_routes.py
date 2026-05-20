"""
Health integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List

try:
    from services.health_integration_service import HealthIntegrationService
except ImportError:
    from ...services.health_integration_service import HealthIntegrationService

router = APIRouter()

health_integration = HealthIntegrationService()


@router.post("/health-app/connect")
async def connect_health_app(
    user_id: str = Body(...),
    app_type: str = Body(...),
    access_token: str = Body(...)
):
    """Conecta una app de salud"""
    try:
        connection = health_integration.connect_health_app(user_id, app_type, access_token)
        return JSONResponse(content=connection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando app: {str(e)}")


@router.post("/health-app/sync")
async def sync_health_data(
    user_id: str = Body(...),
    app_type: str = Body(...),
    data_types: List[str] = Body(...)
):
    """Sincroniza datos de salud"""
    try:
        sync_result = health_integration.sync_health_data(user_id, app_type, data_types)
        return JSONResponse(content=sync_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sincronizando datos: {str(e)}")



