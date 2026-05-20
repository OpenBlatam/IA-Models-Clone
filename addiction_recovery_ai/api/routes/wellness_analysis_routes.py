"""
Wellness analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.wellness_analysis_service import WellnessAnalysisService
except ImportError:
    from ...services.wellness_analysis_service import WellnessAnalysisService

router = APIRouter()

wellness_analysis = WellnessAnalysisService()


@router.post("/wellness/calculate-score")
async def calculate_wellness_score(
    user_id: str = Body(...),
    metrics: Dict = Body(...)
):
    """Calcula puntuación de bienestar general"""
    try:
        score = wellness_analysis.calculate_wellness_score(user_id, metrics)
        return JSONResponse(content=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando bienestar: {str(e)}")


@router.post("/wellness/trends")
async def analyze_wellness_trends(
    user_id: str = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Analiza tendencias de bienestar"""
    try:
        trends = wellness_analysis.analyze_wellness_trends(user_id, historical_data)
        return JSONResponse(content=trends)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando tendencias: {str(e)}")



