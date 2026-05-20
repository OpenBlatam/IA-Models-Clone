"""
Telemedicine integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.telemedicine_integration_service import TelemedicineIntegrationService
except ImportError:
    from ...services.telemedicine_integration_service import TelemedicineIntegrationService

router = APIRouter()

telemedicine = TelemedicineIntegrationService()


@router.post("/telemedicine/schedule-session")
async def schedule_telemedicine_session(
    user_id: str = Body(...),
    provider: str = Body(...),
    session_type: str = Body(...),
    scheduled_time: str = Body(...)
):
    """Programa sesión de telemedicina"""
    try:
        session = telemedicine.schedule_telemedicine_session(
            user_id, provider, session_type, scheduled_time
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error programando sesión: {str(e)}")


@router.get("/telemedicine/available-providers/{user_id}")
async def get_available_providers(
    user_id: str,
    specialty: Optional[str] = Query(None)
):
    """Obtiene proveedores disponibles"""
    try:
        providers = telemedicine.get_available_providers(user_id, specialty)
        return JSONResponse(content={
            "user_id": user_id,
            "providers": providers,
            "total": len(providers),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo proveedores: {str(e)}")



