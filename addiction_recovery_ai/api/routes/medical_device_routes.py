"""
Medical device integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.medical_device_integration_service import MedicalDeviceIntegrationService
except ImportError:
    from ...services.medical_device_integration_service import MedicalDeviceIntegrationService

router = APIRouter()

medical_devices = MedicalDeviceIntegrationService()


@router.post("/medical-devices/register")
async def register_medical_device(
    user_id: str = Body(...),
    device_type: str = Body(...),
    device_id: str = Body(...),
    device_info: Dict = Body(...)
):
    """Registra dispositivo médico"""
    try:
        device = medical_devices.register_medical_device(
            user_id, device_type, device_id, device_info
        )
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


@router.post("/medical-devices/analyze-data")
async def analyze_medical_device_data(
    user_id: str = Body(...),
    device_type: str = Body(...),
    measurements: List[Dict] = Body(...)
):
    """Analiza datos de dispositivo médico"""
    try:
        analysis = medical_devices.analyze_medical_device_data(
            user_id, device_type, measurements
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando datos: {str(e)}")



