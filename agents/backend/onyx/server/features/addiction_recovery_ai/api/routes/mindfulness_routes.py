"""
Mindfulness routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.mindfulness_service import MindfulnessService
except ImportError:
    from ...services.mindfulness_service import MindfulnessService

router = APIRouter()

mindfulness = MindfulnessService()


@router.post("/mindfulness/start-session")
async def start_meditation_session(
    user_id: str = Body(...),
    meditation_type: str = Body(...),
    duration_minutes: int = Body(10)
):
    """Inicia sesión de meditación"""
    try:
        session = mindfulness.start_meditation_session(user_id, meditation_type, duration_minutes)
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando sesión: {str(e)}")


@router.get("/mindfulness/programs/{user_id}")
async def get_meditation_programs(
    user_id: str,
    difficulty: Optional[str] = Query(None)
):
    """Obtiene programas de meditación"""
    try:
        programs = mindfulness.get_meditation_programs(user_id, difficulty)
        return JSONResponse(content={
            "user_id": user_id,
            "programs": programs,
            "total": len(programs),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo programas: {str(e)}")



