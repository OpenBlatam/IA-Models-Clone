"""
Advanced relapse tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_relapse_tracking_service import AdvancedRelapseTrackingService
except ImportError:
    from ...services.advanced_relapse_tracking_service import AdvancedRelapseTrackingService

router = APIRouter()

relapse_tracking = AdvancedRelapseTrackingService()


@router.post("/relapse/record")
async def record_relapse(
    user_id: str = Body(...),
    relapse_data: Dict = Body(...)
):
    """Registra recaída"""
    try:
        relapse = relapse_tracking.record_relapse(user_id, relapse_data)
        return JSONResponse(content=relapse)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando recaída: {str(e)}")


@router.post("/relapse/predict-risk")
async def predict_relapse_risk(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    relapse_history: List[Dict] = Body(...)
):
    """Predice riesgo de recaída"""
    try:
        prediction = relapse_tracking.predict_relapse_risk(
            user_id, current_state, relapse_history
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo riesgo: {str(e)}")



