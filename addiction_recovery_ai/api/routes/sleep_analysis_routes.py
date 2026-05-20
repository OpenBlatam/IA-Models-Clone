"""
Sleep analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.sleep_analysis_service import SleepAnalysisService
except ImportError:
    from ...services.sleep_analysis_service import SleepAnalysisService

router = APIRouter()

sleep_analysis = SleepAnalysisService()


@router.post("/sleep/analyze-patterns")
async def analyze_sleep_patterns(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...)
):
    """Analiza patrones de sueño"""
    try:
        analysis = sleep_analysis.analyze_sleep_patterns(user_id, sleep_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando sueño: {str(e)}")


@router.post("/sleep/correlate")
async def correlate_sleep_with_recovery(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Correlaciona sueño con recuperación"""
    try:
        correlation = sleep_analysis.correlate_sleep_with_recovery(user_id, sleep_data, recovery_data)
        return JSONResponse(content=correlation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error correlacionando datos: {str(e)}")



