"""
Adherence analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_adherence_analysis_service import AdvancedAdherenceAnalysisService
except ImportError:
    from ...services.advanced_adherence_analysis_service import AdvancedAdherenceAnalysisService

router = APIRouter()

adherence_analysis = AdvancedAdherenceAnalysisService()


@router.post("/adherence/calculate-rate")
async def calculate_adherence_rate(
    user_id: str = Body(...),
    expected_actions: List[Dict] = Body(...),
    completed_actions: List[Dict] = Body(...),
    period_days: int = Body(30)
):
    """Calcula tasa de adherencia"""
    try:
        analysis = adherence_analysis.calculate_adherence_rate(
            user_id, expected_actions, completed_actions, period_days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando adherencia: {str(e)}")



