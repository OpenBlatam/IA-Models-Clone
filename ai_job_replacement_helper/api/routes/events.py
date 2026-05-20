"""
Events endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.events import EventsService, EventType

router = APIRouter()
events_service = EventsService()


@router.post("/create/{organizer_id}")
async def create_event(
    organizer_id: str,
    title: str,
    description: str,
    event_type: str,
    start_date: str,
    end_date: str,
    max_participants: Optional[int] = None,
    tags: Optional[str] = None,
    meeting_url: Optional[str] = None
) -> Dict[str, Any]:
    """Crear nuevo evento"""
    try:
        event_type_enum = EventType(event_type)
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        tags_list = tags.split(",") if tags else None
        
        event = events_service.create_event(
            organizer_id, title, description, event_type_enum,
            start_dt, end_dt, max_participants, tags_list, meeting_url
        )
        
        return {
            "id": event.id,
            "title": event.title,
            "start_date": event.start_date.isoformat(),
            "status": event.status.value,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upcoming")
async def get_upcoming_events(
    event_type: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Obtener eventos próximos"""
    try:
        event_type_enum = EventType(event_type) if event_type else None
        events = events_service.get_upcoming_events(event_type_enum, limit)
        return {
            "events": [
                {
                    "id": e.id,
                    "title": e.title,
                    "type": e.event_type.value,
                    "start_date": e.start_date.isoformat(),
                    "registered_count": len(e.registered_participants),
                    "max_participants": e.max_participants,
                }
                for e in events
            ],
            "total": len(events),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register/{user_id}/{event_id}")
async def register_for_event(user_id: str, event_id: str) -> Dict[str, Any]:
    """Registrarse para un evento"""
    try:
        success = events_service.register_for_event(user_id, event_id)
        if not success:
            raise HTTPException(status_code=400, detail="Registration failed")
        return {"success": True, "event_id": event_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_events(user_id: str) -> Dict[str, Any]:
    """Obtener eventos del usuario"""
    try:
        events = events_service.get_user_events(user_id)
        return {
            "organized": [
                {
                    "id": e.id,
                    "title": e.title,
                    "start_date": e.start_date.isoformat(),
                }
                for e in events["organized"]
            ],
            "registered": [
                {
                    "id": e.id,
                    "title": e.title,
                    "start_date": e.start_date.isoformat(),
                }
                for e in events["registered"]
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




