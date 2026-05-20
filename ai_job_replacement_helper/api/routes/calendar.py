"""
Calendar endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.calendar_integration import CalendarIntegrationService

router = APIRouter()
calendar_service = CalendarIntegrationService()


@router.post("/event/{user_id}")
async def create_event(
    user_id: str,
    title: str,
    description: str,
    start_time: str,
    end_time: str,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """Crear evento en calendario"""
    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        
        event = calendar_service.create_event(
            user_id, title, description, start_dt, end_dt, location
        )
        
        return {
            "id": event.id,
            "title": event.title,
            "start_time": event.start_time.isoformat(),
            "end_time": event.end_time.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upcoming/{user_id}")
async def get_upcoming_events(
    user_id: str,
    days: int = 7
) -> Dict[str, Any]:
    """Obtener eventos próximos"""
    try:
        events = calendar_service.get_upcoming_events(user_id, days)
        return {
            "events": [
                {
                    "id": e.id,
                    "title": e.title,
                    "start_time": e.start_time.isoformat(),
                    "end_time": e.end_time.isoformat(),
                }
                for e in events
            ],
            "total": len(events),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export-ical/{user_id}")
async def export_ical(user_id: str) -> Dict[str, Any]:
    """Exportar eventos a iCal"""
    try:
        ical_content = calendar_service.export_to_ical(user_id)
        return {
            "content": ical_content,
            "format": "text/calendar",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




