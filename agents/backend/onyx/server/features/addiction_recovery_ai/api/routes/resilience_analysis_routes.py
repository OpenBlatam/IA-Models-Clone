"""
Resilience analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.resilience_analysis_service import ResilienceAnalysisService
except ImportError:
    from ...services.resilience_analysis_service import ResilienceAnalysisService

router = APIRouter()

resilience_analysis = ResilienceAnalysisService()


@router.post("/resilience/assess")
async def assess_resilience(
    user_id: str = Body(...),
    resilience_data: Dict = Body(...)
):
    """Evalúa resiliencia"""
    try:
        assessment = resilience_analysis.assess_resilience(user_id, resilience_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando resiliencia: {str(e)}")


@router.post("/resilience/predict-outcome")
async def predict_resilience_outcome(
    user_id: str = Body(...),
    current_resilience: Dict = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Predice resultado de resiliencia"""
    try:
        prediction = resilience_analysis.predict_resilience_outcome(
            user_id, current_resilience, historical_data
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo resultado: {str(e)}")



