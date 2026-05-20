"""
Predictive AI routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.predictive_ai_service import PredictiveAIService
except ImportError:
    from ...services.predictive_ai_service import PredictiveAIService

router = APIRouter()

predictive_ai = PredictiveAIService()


@router.post("/predictive/success-probability")
async def predict_success_probability(
    user_id: str = Body(...),
    days_sober: int = Body(...),
    historical_data: Dict = Body(...)
):
    """Predice probabilidad de éxito en recuperación"""
    try:
        prediction = predictive_ai.predict_success_probability(user_id, days_sober, historical_data)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo probabilidad: {str(e)}")


@router.post("/predictive/relapse-window")
async def predict_relapse_window(
    user_id: str = Body(...),
    current_data: Dict = Body(...)
):
    """Predice ventana de riesgo de recaída"""
    try:
        prediction = predictive_ai.predict_relapse_window(user_id, current_data)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo ventana: {str(e)}")



