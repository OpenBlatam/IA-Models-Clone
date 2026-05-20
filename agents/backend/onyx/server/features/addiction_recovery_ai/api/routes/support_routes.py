"""
Support and motivation routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

try:
    from services.counseling_service import CounselingService
    from services.motivation_service import MotivationService
except ImportError:
    from ...services.counseling_service import CounselingService
    from ...services.motivation_service import MotivationService

router = APIRouter()

counseling = CounselingService()
motivation = MotivationService()


class CoachingSessionRequest(BaseModel):
    user_id: str
    topic: str
    current_situation: str
    questions: Optional[List[str]] = None


@router.post("/coaching-session")
async def coaching_session(request: CoachingSessionRequest):
    """Sesión de coaching personalizado"""
    try:
        session = counseling.create_coaching_session(
            request.user_id,
            request.topic,
            request.current_situation,
            request.questions
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en sesión de coaching: {str(e)}")


@router.get("/motivation/{user_id}")
async def get_motivation(user_id: str):
    """Obtiene mensajes motivacionales personalizados"""
    try:
        messages = motivation.get_motivational_messages(user_id, {})
        return JSONResponse(content=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo motivación: {str(e)}")


@router.post("/celebrate-milestone")
async def celebrate_milestone(
    user_id: str = Body(...),
    milestone_days: int = Body(...)
):
    """Celebra un logro/hito"""
    try:
        celebration = motivation.celebrate_milestone(user_id, milestone_days)
        return JSONResponse(content=celebration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error celebrando hito: {str(e)}")


@router.get("/achievements/{user_id}")
async def get_achievements(user_id: str):
    """Obtiene logros del usuario"""
    try:
        achievements = motivation.get_achievements(user_id)
        return JSONResponse(content=achievements)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo logros: {str(e)}")



