"""
Emergency integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.emergency_integration_service import EmergencyIntegrationService
except ImportError:
    from ...services.emergency_integration_service import EmergencyIntegrationService

router = APIRouter()

emergency_integration = EmergencyIntegrationService()


@router.get("/emergency/services")
async def get_emergency_services(
    location: Optional[str] = Query(None),
    service_type: Optional[str] = Query(None)
):
    """Obtiene servicios de emergencia disponibles"""
    try:
        services = emergency_integration.get_emergency_services(location, service_type)
        return JSONResponse(content={
            "services": services,
            "location": location,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo servicios: {str(e)}")


@router.get("/emergency/crisis-resources")
async def get_crisis_resources(location: Optional[str] = Query(None)):
    """Obtiene recursos de crisis"""
    try:
        resources = emergency_integration.get_crisis_resources(location)
        return JSONResponse(content=resources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recursos: {str(e)}")



