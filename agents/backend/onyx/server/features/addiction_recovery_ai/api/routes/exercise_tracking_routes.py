"""
Exercise tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_exercise_tracking_service import AdvancedExerciseTrackingService
except ImportError:
    from ...services.advanced_exercise_tracking_service import AdvancedExerciseTrackingService

router = APIRouter()

advanced_exercise = AdvancedExerciseTrackingService()


@router.post("/exercise/record-session")
async def record_exercise_session(
    user_id: str = Body(...),
    exercise_data: Dict = Body(...)
):
    """Registra sesión de ejercicio"""
    try:
        session = advanced_exercise.record_exercise_session(user_id, exercise_data)
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando sesión: {str(e)}")


@router.post("/exercise/analyze-patterns")
async def analyze_exercise_patterns(
    user_id: str = Body(...),
    exercise_sessions: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones de ejercicio"""
    try:
        analysis = advanced_exercise.analyze_exercise_patterns(
            user_id, exercise_sessions, days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")



