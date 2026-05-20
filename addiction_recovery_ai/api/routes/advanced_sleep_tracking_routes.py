"""
Advanced sleep tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional

try:
    from services.advanced_sleep_tracking_service import AdvancedSleepTrackingService
except ImportError:
    from ...services.advanced_sleep_tracking_service import AdvancedSleepTrackingService

router = APIRouter()

advanced_sleep = AdvancedSleepTrackingService()


@router.post("/sleep/advanced/record")
async def record_sleep_data_advanced(
    user_id: str = Body(...),
    sleep_start: str = Body(...),
    sleep_end: str = Body(...),
    sleep_stages: Optional[Dict] = Body(None)
):
    """Registra datos de sueño avanzado"""
    try:
        sleep_record = advanced_sleep.record_sleep_data(
            user_id, sleep_start, sleep_end, sleep_stages
        )
        return JSONResponse(content=sleep_record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando sueño: {str(e)}")


@router.post("/sleep/advanced/analyze-patterns")
async def analyze_sleep_patterns_advanced(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones de sueño avanzado"""
    try:
        analysis = advanced_sleep.analyze_sleep_patterns_advanced(user_id, sleep_data, days)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")



