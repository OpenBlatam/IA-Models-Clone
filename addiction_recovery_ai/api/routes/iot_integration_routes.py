"""
IoT integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.iot_integration_service import IoTIntegrationService
except ImportError:
    from ...services.iot_integration_service import IoTIntegrationService

router = APIRouter()

iot_integration = IoTIntegrationService()


@router.post("/iot/register-device")
async def register_iot_device(
    user_id: str = Body(...),
    device_type: str = Body(...),
    device_name: str = Body(...),
    device_id: str = Body(...),
    connection_info: Dict = Body(...)
):
    """Registra dispositivo IoT"""
    try:
        device = iot_integration.register_iot_device(
            user_id, device_type, device_name, device_id, connection_info
        )
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


@router.get("/iot/health-metrics/{user_id}")
async def get_iot_health_metrics(
    user_id: str,
    device_type: Optional[str] = Query(None),
    days: int = Query(7)
):
    """Obtiene métricas de salud de dispositivos IoT"""
    try:
        metrics = iot_integration.get_iot_health_metrics(user_id, device_type, days)
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas: {str(e)}")



