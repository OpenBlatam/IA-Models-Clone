"""
Long term success prediction routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.long_term_success_prediction_service import LongTermSuccessPredictionService
except ImportError:
    from ...services.long_term_success_prediction_service import LongTermSuccessPredictionService

router = APIRouter()

long_term_prediction = LongTermSuccessPredictionService()


@router.post("/prediction/long-term-success")
async def predict_long_term_success(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    historical_data: List[Dict] = Body(...),
    prediction_horizon_years: int = Body(5)
):
    """Predice éxito a largo plazo"""
    try:
        prediction = long_term_prediction.predict_long_term_success(
            user_id, current_state, historical_data, prediction_horizon_years
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo éxito: {str(e)}")


@router.post("/prediction/milestone-achievement")
async def predict_milestone_achievement(
    user_id: str = Body(...),
    milestone: str = Body(...),
    current_progress: Dict = Body(...)
):
    """Predice logro de hito"""
    try:
        prediction = long_term_prediction.predict_milestone_achievement(
            user_id, milestone, current_progress
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo hito: {str(e)}")



