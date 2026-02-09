"""
Emergency services integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.emergency_services_integration_service import EmergencyServicesIntegrationService
except ImportError:
    from ...services.emergency_services_integration_service import EmergencyServicesIntegrationService

router = APIRouter()

emergency_services = EmergencyServicesIntegrationService()


@router.post("/emergency/trigger")
async def trigger_emergency(
    user_id: str = Body(...),
    emergency_type: str = Body(...),
    emergency_data: Dict = Body(...)
):
    """Activa emergencia"""
    try:
        emergency = emergency_services.trigger_emergency(
            user_id, emergency_type, emergency_data
        )
        return JSONResponse(content=emergency)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activando emergencia: {str(e)}")


@router.post("/emergency/resources")
async def get_emergency_resources(
    user_id: str = Body(...),
    location: Optional[Dict] = Body(None)
):
    """Obtiene recursos de emergencia"""
    try:
        resources = emergency_services.get_emergency_resources(user_id, location)
        return JSONResponse(content=resources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recursos: {str(e)}")



