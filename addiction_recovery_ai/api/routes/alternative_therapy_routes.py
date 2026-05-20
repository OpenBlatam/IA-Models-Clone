"""
Alternative therapy routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.alternative_therapy_integration_service import AlternativeTherapyIntegrationService
except ImportError:
    from ...services.alternative_therapy_integration_service import AlternativeTherapyIntegrationService

router = APIRouter()

alternative_therapy = AlternativeTherapyIntegrationService()


@router.post("/alternative-therapy/recommend")
async def recommend_therapy(
    user_id: str = Body(...),
    user_profile: Dict = Body(...),
    current_state: Dict = Body(...)
):
    """Recomienda terapia alternativa"""
    try:
        recommendation = alternative_therapy.recommend_therapy(
            user_id, user_profile, current_state
        )
        return JSONResponse(content=recommendation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recomendando terapia: {str(e)}")


@router.post("/alternative-therapy/track-session")
async def track_therapy_session(
    user_id: str = Body(...),
    therapy_type: str = Body(...),
    session_data: Dict = Body(...)
):
    """Rastrea sesión de terapia"""
    try:
        session = alternative_therapy.track_therapy_session(
            user_id, therapy_type, session_data
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando sesión: {str(e)}")



