"""
Advanced biometrics routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_biometrics_service import AdvancedBiometricsService
except ImportError:
    from ...services.advanced_biometrics_service import AdvancedBiometricsService

router = APIRouter()

advanced_biometrics = AdvancedBiometricsService()


@router.post("/biometrics/record")
async def record_biometric_data(
    user_id: str = Body(...),
    biometric_type: str = Body(...),
    measurements: Dict = Body(...)
):
    """Registra datos biométricos"""
    try:
        record = advanced_biometrics.record_biometric_data(
            user_id, biometric_type, measurements
        )
        return JSONResponse(content=record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando datos: {str(e)}")


@router.post("/biometrics/analyze-trends")
async def analyze_biometric_trends(
    user_id: str = Body(...),
    biometric_data: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza tendencias biométricas"""
    try:
        analysis = advanced_biometrics.analyze_biometric_trends(
            user_id, biometric_data, days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando tendencias: {str(e)}")



