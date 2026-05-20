"""
Long term progress analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.long_term_progress_analysis_service import LongTermProgressAnalysisService
except ImportError:
    from ...services.long_term_progress_analysis_service import LongTermProgressAnalysisService

router = APIRouter()

long_term_progress = LongTermProgressAnalysisService()


@router.post("/long-term-progress/analyze")
async def analyze_long_term_progress(
    user_id: str = Body(...),
    progress_data: List[Dict] = Body(...),
    time_period_months: int = Body(12)
):
    """Analiza progreso a largo plazo"""
    try:
        analysis = long_term_progress.analyze_long_term_progress(
            user_id, progress_data, time_period_months
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando progreso: {str(e)}")


@router.post("/long-term-progress/predict-outcome")
async def predict_long_term_outcome(
    user_id: str = Body(...),
    current_progress: Dict = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Predice resultado a largo plazo"""
    try:
        prediction = long_term_progress.predict_long_term_outcome(
            user_id, current_progress, historical_data
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo resultado: {str(e)}")



