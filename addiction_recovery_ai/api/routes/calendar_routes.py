"""
Calendar routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from datetime import datetime

try:
    from services.calendar_service import CalendarService
except ImportError:
    from ...services.calendar_service import CalendarService

router = APIRouter()

calendar = CalendarService()


@router.post("/calendar/event")
async def create_calendar_event(
    user_id: str = Body(...),
    event_type: str = Body(...),
    title: str = Body(...),
    description: str = Body(...),
    scheduled_time: str = Body(...),
    repeat_daily: bool = Body(False),
    repeat_weekly: bool = Body(False),
    reminder_minutes: int = Body(15)
):
    """Crea un evento en el calendario"""
    try:
        scheduled = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
        event = calendar.create_event(
            user_id, event_type, title, description, scheduled,
            repeat_daily, repeat_weekly, reminder_minutes
        )
        return JSONResponse(content=event)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando evento: {str(e)}")


@router.get("/calendar/upcoming/{user_id}")
async def get_upcoming_events(
    user_id: str,
    days_ahead: int = Query(7, ge=1, le=30)
):
    """Obtiene eventos próximos del usuario"""
    try:
        events = calendar.get_upcoming_events(user_id, days_ahead)
        return JSONResponse(content={
            "user_id": user_id,
            "events": events,
            "days_ahead": days_ahead,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo eventos: {str(e)}")


@router.post("/calendar/daily-reminders/{user_id}")
async def create_daily_reminders(user_id: str):
    """Crea recordatorios diarios automáticos"""
    try:
        reminders = calendar.create_daily_reminders(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "reminders_created": len(reminders),
            "reminders": reminders,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando recordatorios: {str(e)}")



