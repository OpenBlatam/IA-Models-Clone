"""
Comparative analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.comparative_progress_analysis_service import ComparativeProgressAnalysisService
except ImportError:
    from ...services.comparative_progress_analysis_service import ComparativeProgressAnalysisService

router = APIRouter()

comparative_analysis = ComparativeProgressAnalysisService()


@router.post("/comparative/compare-periods")
async def compare_periods(
    user_id: str = Body(...),
    period1_data: List[Dict] = Body(...),
    period2_data: List[Dict] = Body(...),
    metrics: List[str] = Body(...)
):
    """Compara dos períodos"""
    try:
        comparison = comparative_analysis.compare_periods(
            user_id, period1_data, period2_data, metrics
        )
        return JSONResponse(content=comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparando períodos: {str(e)}")


@router.post("/comparative/compare-with-baseline")
async def compare_with_baseline(
    user_id: str = Body(...),
    current_data: Dict = Body(...),
    baseline_data: Dict = Body(...),
    metrics: List[str] = Body(...)
):
    """Compara con línea base"""
    try:
        comparison = comparative_analysis.compare_with_baseline(
            user_id, current_data, baseline_data, metrics
        )
        return JSONResponse(content=comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparando con línea base: {str(e)}")



