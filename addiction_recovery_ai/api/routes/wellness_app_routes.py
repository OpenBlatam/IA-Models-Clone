"""
Wellness app integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.wellness_app_integration_service import WellnessAppIntegrationService
except ImportError:
    from ...services.wellness_app_integration_service import WellnessAppIntegrationService

router = APIRouter()

wellness_apps = WellnessAppIntegrationService()


@router.post("/wellness/connect-app")
async def connect_wellness_app(
    user_id: str = Body(...),
    app_type: str = Body(...),
    connection_info: Dict = Body(...)
):
    """Conecta app de bienestar"""
    try:
        connection = wellness_apps.connect_wellness_app(
            user_id, app_type, connection_info
        )
        return JSONResponse(content=connection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando app: {str(e)}")


@router.post("/wellness/analyze-impact")
async def analyze_wellness_impact(
    user_id: str = Body(...),
    wellness_data: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Analiza impacto de bienestar en recuperación"""
    try:
        analysis = wellness_apps.analyze_wellness_impact(
            user_id, wellness_data, recovery_data
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando impacto: {str(e)}")



