"""
Advanced intelligent reminders routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.advanced_intelligent_reminders_service import AdvancedIntelligentRemindersService
except ImportError:
    from ...services.advanced_intelligent_reminders_service import AdvancedIntelligentRemindersService

router = APIRouter()

intelligent_reminders = AdvancedIntelligentRemindersService()


@router.post("/reminders/create-intelligent")
async def create_intelligent_reminder(
    user_id: str = Body(...),
    reminder_data: Dict = Body(...)
):
    """Crea recordatorio inteligente"""
    try:
        reminder = intelligent_reminders.create_intelligent_reminder(user_id, reminder_data)
        return JSONResponse(content=reminder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando recordatorio: {str(e)}")


@router.post("/reminders/optimize-timing")
async def optimize_reminder_timing(
    user_id: str = Body(...),
    reminder_type: str = Body(...),
    user_patterns: Dict = Body(...)
):
    """Optimiza horario de recordatorio"""
    try:
        optimization = intelligent_reminders.optimize_reminder_timing(
            user_id, reminder_type, user_patterns
        )
        return JSONResponse(content=optimization)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizando horario: {str(e)}")



