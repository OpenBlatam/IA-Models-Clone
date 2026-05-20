"""
Meditation app integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.meditation_app_integration_service import MeditationAppIntegrationService
except ImportError:
    from ...services.meditation_app_integration_service import MeditationAppIntegrationService

router = APIRouter()

meditation_apps = MeditationAppIntegrationService()


@router.post("/meditation/connect-app")
async def connect_meditation_app(
    user_id: str = Body(...),
    app_type: str = Body(...),
    connection_info: Dict = Body(...)
):
    """Conecta app de meditación"""
    try:
        connection = meditation_apps.connect_meditation_app(
            user_id, app_type, connection_info
        )
        return JSONResponse(content=connection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando app: {str(e)}")


@router.post("/meditation/analyze-impact")
async def analyze_meditation_impact(
    user_id: str = Body(...),
    meditation_data: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Analiza impacto de meditación en recuperación"""
    try:
        analysis = meditation_apps.analyze_meditation_impact(
            user_id, meditation_data, recovery_data
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando impacto: {str(e)}")



