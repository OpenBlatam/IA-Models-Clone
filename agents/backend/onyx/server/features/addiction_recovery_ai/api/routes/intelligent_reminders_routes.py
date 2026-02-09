"""
Intelligent reminders routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse

try:
    from services.intelligent_reminders_service import IntelligentRemindersService
except ImportError:
    from ...services.intelligent_reminders_service import IntelligentRemindersService

router = APIRouter()

intelligent_reminders = IntelligentRemindersService()


@router.post("/reminders/create")
async def create_reminder(
    user_id: str = Body(...),
    reminder_type: str = Body(...),
    title: str = Body(...),
    message: str = Body(...),
    scheduled_time: str = Body(...)
):
    """Crea un recordatorio"""
    try:
        reminder = intelligent_reminders.create_reminder(
            user_id, reminder_type, title, message, scheduled_time
        )
        return JSONResponse(content=reminder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando recordatorio: {str(e)}")


@router.get("/reminders/upcoming/{user_id}")
async def get_upcoming_reminders(user_id: str, hours_ahead: int = Query(24)):
    """Obtiene recordatorios próximos"""
    try:
        reminders = intelligent_reminders.get_upcoming_reminders(user_id, hours_ahead)
        return JSONResponse(content={
            "user_id": user_id,
            "reminders": reminders,
            "total": len(reminders),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recordatorios: {str(e)}")



