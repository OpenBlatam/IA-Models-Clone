"""
Health monitoring devices routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.health_monitoring_device_service import HealthMonitoringDeviceService
except ImportError:
    from ...services.health_monitoring_device_service import HealthMonitoringDeviceService

router = APIRouter()

health_devices = HealthMonitoringDeviceService()


@router.post("/health-devices/register")
async def register_health_device(
    user_id: str = Body(...),
    device_type: str = Body(...),
    device_id: str = Body(...),
    device_info: Dict = Body(...)
):
    """Registra dispositivo de salud"""
    try:
        device = health_devices.register_health_device(
            user_id, device_type, device_id, device_info
        )
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


@router.post("/health-devices/record-reading")
async def record_health_reading(
    user_id: str = Body(...),
    device_id: str = Body(...),
    reading_data: Dict = Body(...)
):
    """Registra lectura de salud"""
    try:
        reading = health_devices.record_health_reading(
            user_id, device_id, reading_data
        )
        return JSONResponse(content=reading)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando lectura: {str(e)}")



