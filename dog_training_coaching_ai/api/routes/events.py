"""
Events Endpoint
===============
Endpoint para sistema de eventos.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...utils.event_system import get_event_bus, EventType, Event
from ...utils.logger import get_logger

router = APIRouter(prefix="/api/v1/events", tags=["events"])
logger = get_logger(__name__)


@router.get("/stats")
async def get_event_stats() -> Dict[str, Any]:
    """
    Obtener estadísticas del sistema de eventos.
    
    Returns:
        Estadísticas de eventos
    """
    try:
        event_bus = get_event_bus()
        return {
            "success": True,
            "stats": event_bus.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting event stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_event_history(
    event_type: Optional[str] = Query(None, description="Filtrar por tipo de evento"),
    limit: int = Query(100, ge=1, le=1000, description="Límite de eventos")
) -> Dict[str, Any]:
    """
    Obtener historial de eventos.
    
    Returns:
        Historial de eventos
    """
    try:
        event_bus = get_event_bus()
        
        filter_type = None
        if event_type:
            try:
                filter_type = EventType(event_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        
        history = event_bus.get_history(event_type=filter_type, limit=limit)
        
        return {
            "success": True,
            "events": [
                {
                    "id": event.id,
                    "type": event.event_type.value,
                    "data": event.data,
                    "source": event.source,
                    "timestamp": event.timestamp.isoformat()
                }
                for event in history
            ],
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting event history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish")
async def publish_event(
    event_type: str,
    data: Dict[str, Any],
    source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Publicar un evento manualmente.
    
    Returns:
        Confirmación de publicación
    """
    try:
        try:
            event_type_enum = EventType(event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        
        event_bus = get_event_bus()
        event = Event(
            event_type=event_type_enum,
            data=data,
            source=source or "api"
        )
        
        await event_bus.publish(event)
        
        return {
            "success": True,
            "event_id": event.id,
            "message": "Event published successfully",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

