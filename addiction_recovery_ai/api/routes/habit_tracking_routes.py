"""
Habit tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.habit_tracking_service import HabitTrackingService
except ImportError:
    from ...services.habit_tracking_service import HabitTrackingService

router = APIRouter()

habit_tracking = HabitTrackingService()


@router.post("/habits/create")
async def create_habit(
    user_id: str = Body(...),
    name: str = Body(...),
    description: str = Body(...),
    frequency: str = Body("daily")
):
    """Crea un nuevo hábito"""
    try:
        habit = habit_tracking.create_habit(user_id, name, description, frequency)
        return JSONResponse(content=habit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando hábito: {str(e)}")


@router.post("/habits/log-completion")
async def log_habit_completion(
    habit_id: str = Body(...),
    user_id: str = Body(...),
    value: Optional[float] = Body(None)
):
    """Registra completación de hábito"""
    try:
        completion = habit_tracking.log_habit_completion(habit_id, user_id, value)
        return JSONResponse(content=completion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando completación: {str(e)}")



