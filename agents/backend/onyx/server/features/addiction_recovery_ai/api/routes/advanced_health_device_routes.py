"""
Advanced health device integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_health_device_integration_service import AdvancedHealthDeviceIntegrationService
except ImportError:
    from ...services.advanced_health_device_integration_service import AdvancedHealthDeviceIntegrationService

router = APIRouter()

health_devices_advanced = AdvancedHealthDeviceIntegrationService()


@router.post("/health-devices-advanced/register")
async def register_health_device_advanced(
    user_id: str = Body(...),
    device_data: Dict = Body(...)
):
    """Registra dispositivo de salud avanzado"""
    try:
        device = health_devices_advanced.register_device(user_id, device_data)
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


@router.post("/health-devices-advanced/analyze-data")
async def analyze_device_data(
    user_id: str = Body(...),
    device_id: str = Body(...),
    data_points: List[Dict] = Body(...)
):
    """Analiza datos del dispositivo"""
    try:
        analysis = health_devices_advanced.analyze_device_data(
            user_id, device_id, data_points
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando datos: {str(e)}")



