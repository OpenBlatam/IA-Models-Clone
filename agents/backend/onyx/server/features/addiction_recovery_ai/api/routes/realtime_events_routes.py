"""
Realtime events routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.realtime_events_service import RealtimeEventsService
except ImportError:
    from ...services.realtime_events_service import RealtimeEventsService

router = APIRouter()

realtime_events = RealtimeEventsService()


@router.post("/events/log")
async def log_event(
    user_id: str = Body(...),
    event_type: str = Body(...),
    event_data: Dict = Body(...)
):
    """Registra un evento"""
    try:
        event = realtime_events.log_event(user_id, event_type, event_data)
        return JSONResponse(content=event)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando evento: {str(e)}")


@router.get("/events/recent/{user_id}")
async def get_recent_events(
    user_id: str,
    event_types: Optional[str] = Query(None),
    limit: int = Query(50)
):
    """Obtiene eventos recientes"""
    try:
        event_types_list = event_types.split(",") if event_types else None
        events = realtime_events.get_recent_events(user_id, event_types_list, limit)
        return JSONResponse(content={
            "user_id": user_id,
            "events": events,
            "total": len(events),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo eventos: {str(e)}")



