"""
Advanced therapy tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_therapy_tracking_service import AdvancedTherapyTrackingService
except ImportError:
    from ...services.advanced_therapy_tracking_service import AdvancedTherapyTrackingService

router = APIRouter()

therapy_tracking = AdvancedTherapyTrackingService()


@router.post("/therapy/track-session")
async def track_therapy_session(
    user_id: str = Body(...),
    session_data: Dict = Body(...)
):
    """Rastrea sesión de terapia"""
    try:
        session = therapy_tracking.track_therapy_session(user_id, session_data)
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando sesión: {str(e)}")


@router.post("/therapy/recommend-adjustments")
async def recommend_therapy_adjustments(
    user_id: str = Body(...),
    current_therapy: Dict = Body(...),
    progress_data: List[Dict] = Body(...)
):
    """Recomienda ajustes de terapia"""
    try:
        recommendations = therapy_tracking.recommend_therapy_adjustments(
            user_id, current_therapy, progress_data
        )
        return JSONResponse(content=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recomendando ajustes: {str(e)}")



