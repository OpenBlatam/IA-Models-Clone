"""
Advanced predictive ML routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.advanced_predictive_ml_service import AdvancedPredictiveMLService
except ImportError:
    from ...services.advanced_predictive_ml_service import AdvancedPredictiveMLService

router = APIRouter()

advanced_predictive_ml = AdvancedPredictiveMLService()


@router.post("/ml-predictive/predict-outcome")
async def predict_recovery_outcome(
    user_id: str = Body(...),
    input_features: Dict = Body(...),
    model_version: str = Body("latest")
):
    """Predice resultado de recuperación usando ML"""
    try:
        prediction = advanced_predictive_ml.predict_recovery_outcome(
            user_id, input_features, model_version
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo resultado: {str(e)}")



