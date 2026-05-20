"""
Temporal patterns analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.temporal_pattern_analysis_service import TemporalPatternAnalysisService
except ImportError:
    from ...services.temporal_pattern_analysis_service import TemporalPatternAnalysisService

router = APIRouter()

temporal_patterns = TemporalPatternAnalysisService()


@router.post("/patterns/daily")
async def analyze_daily_patterns(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    metric: str = Body("mood")
):
    """Analiza patrones diarios"""
    try:
        analysis = temporal_patterns.analyze_daily_patterns(user_id, data, metric)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/patterns/weekly")
async def analyze_weekly_patterns(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    metric: str = Body("check_ins")
):
    """Analiza patrones semanales"""
    try:
        analysis = temporal_patterns.analyze_weekly_patterns(user_id, data, metric)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")



