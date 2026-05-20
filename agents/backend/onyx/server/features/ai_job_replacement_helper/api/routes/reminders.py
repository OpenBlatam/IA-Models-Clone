"""
Reminders endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.reminders import RemindersService, ReminderType, ReminderStatus

router = APIRouter()
reminders_service = RemindersService()


@router.post("/create/{user_id}")
async def create_reminder(
    user_id: str,
    reminder_type: str,
    title: str,
    message: str,
    due_date: str,
    recurring: bool = False,
    recurring_interval_days: Optional[int] = None
) -> Dict[str, Any]:
    """Crear recordatorio"""
    try:
        type_enum = ReminderType(reminder_type)
        due_dt = datetime.fromisoformat(due_date)
        
        reminder = reminders_service.create_reminder(
            user_id, type_enum, title, message, due_dt, recurring, recurring_interval_days
        )
        
        return {
            "id": reminder.id,
            "title": reminder.title,
            "due_date": reminder.due_date.isoformat(),
            "status": reminder.status.value,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def get_reminders(
    user_id: str,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """Obtener recordatorios del usuario"""
    try:
        status_enum = ReminderStatus(status) if status else None
        reminders = reminders_service.get_user_reminders(user_id, status_enum)
        
        return {
            "reminders": [
                {
                    "id": r.id,
                    "title": r.title,
                    "message": r.message,
                    "due_date": r.due_date.isoformat(),
                    "status": r.status.value,
                    "type": r.reminder_type.value,
                }
                for r in reminders
            ],
            "total": len(reminders),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/due/{user_id}")
async def get_due_reminders(user_id: str) -> Dict[str, Any]:
    """Obtener recordatorios vencidos"""
    try:
        due = reminders_service.get_due_reminders(user_id)
        return {
            "reminders": [
                {
                    "id": r.id,
                    "title": r.title,
                    "message": r.message,
                    "due_date": r.due_date.isoformat(),
                }
                for r in due
            ],
            "total": len(due),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




