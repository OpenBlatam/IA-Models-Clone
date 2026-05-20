"""
Sleep pattern analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_sleep_pattern_analysis_service import AdvancedSleepPatternAnalysisService
except ImportError:
    from ...services.advanced_sleep_pattern_analysis_service import AdvancedSleepPatternAnalysisService

router = APIRouter()

sleep_patterns = AdvancedSleepPatternAnalysisService()


@router.post("/sleep-patterns/analyze")
async def analyze_sleep_patterns(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...)
):
    """Analiza patrones de sueño"""
    try:
        analysis = sleep_patterns.analyze_sleep_patterns(user_id, sleep_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/sleep-patterns/correlate-with-recovery")
async def correlate_sleep_with_recovery(
    user_id: str = Body(...),
    sleep_data: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Correlaciona sueño con recuperación"""
    try:
        correlation = sleep_patterns.correlate_sleep_with_recovery(
            user_id, sleep_data, recovery_data
        )
        return JSONResponse(content=correlation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error correlacionando: {str(e)}")



