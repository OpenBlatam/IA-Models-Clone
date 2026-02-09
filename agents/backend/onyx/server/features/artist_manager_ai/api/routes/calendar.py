"""
Calendar API Routes
===================

Endpoints para gestión de calendarios.
"""

import os
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

router = APIRouter(prefix="/calendar", tags=["calendar"])


class CalendarEventCreate(BaseModel):
    title: str
    description: str
    event_type: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    attendees: List[str] = None
    protocol_requirements: List[str] = None
    wardrobe_requirements: Optional[str] = None
    notes: Optional[str] = None


class CalendarEventUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    attendees: Optional[List[str]] = None
    notes: Optional[str] = None


def get_artist_manager(artist_id: str):
    """Dependency para obtener ArtistManager."""
    from ...core.artist_manager import ArtistManager
    from ...core.calendar_manager import CalendarEvent, EventType
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    manager = ArtistManager(artist_id=artist_id, openrouter_api_key=openrouter_key)
    return manager, CalendarEvent, EventType


@router.post("/{artist_id}/events", response_model=Dict[str, Any])
async def create_event(artist_id: str, event: CalendarEventCreate):
    """Crear nuevo evento."""
    try:
        manager, CalendarEvent, EventType = get_artist_manager(artist_id)
        
        import uuid
        event_id = str(uuid.uuid4())
        
        event_type = EventType(event.event_type) if event.event_type in [e.value for e in EventType] else EventType.OTHER
        
        calendar_event = CalendarEvent(
            id=event_id,
            title=event.title,
            description=event.description,
            event_type=event_type,
            start_time=event.start_time,
            end_time=event.end_time,
            location=event.location,
            attendees=event.attendees or [],
            protocol_requirements=event.protocol_requirements or [],
            wardrobe_requirements=event.wardrobe_requirements,
            notes=event.notes
        )
        
        created_event = manager.calendar.add_event(calendar_event)
        return created_event.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/events", response_model=List[Dict[str, Any]])
async def get_events(
    artist_id: str,
    date: Optional[datetime] = None,
    days: Optional[int] = None
):
    """Obtener eventos."""
    try:
        manager, _, _ = get_artist_manager(artist_id)
        
        if date:
            events = manager.calendar.get_events_by_date(date)
        elif days:
            events = manager.calendar.get_upcoming_events(days=days)
        else:
            events = manager.calendar.get_all_events()
        
        return [e.to_dict() for e in events]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/events/{event_id}", response_model=Dict[str, Any])
async def get_event(artist_id: str, event_id: str):
    """Obtener evento específico."""
    try:
        manager, _, _ = get_artist_manager(artist_id)
        event = manager.calendar.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        return event.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{artist_id}/events/{event_id}", response_model=Dict[str, Any])
async def update_event(artist_id: str, event_id: str, event_update: CalendarEventUpdate):
    """Actualizar evento."""
    try:
        manager, _, _ = get_artist_manager(artist_id)
        
        updates = event_update.dict(exclude_unset=True)
        updated_event = manager.calendar.update_event(event_id, **updates)
        return updated_event.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{artist_id}/events/{event_id}")
async def delete_event(artist_id: str, event_id: str):
    """Eliminar evento."""
    try:
        manager, _, _ = get_artist_manager(artist_id)
        success = manager.calendar.delete_event(event_id)
        if not success:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"status": "deleted", "event_id": event_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/events/{event_id}/wardrobe-recommendation", response_model=Dict[str, Any])
async def get_wardrobe_recommendation(artist_id: str, event_id: str):
    """Obtener recomendación de vestimenta para evento."""
    try:
        manager, _, _ = get_artist_manager(artist_id)
        recommendation = await manager.generate_wardrobe_recommendation(event_id)
        return recommendation.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




