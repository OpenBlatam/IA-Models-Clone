"""
Streaming API Endpoints
=======================

Endpoints para streaming y event bus.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
import logging

from ..core.streaming_system import (
    get_streaming_system,
    StreamStatus
)
from ..core.event_bus import get_event_bus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/streaming", tags=["streaming"])


@router.post("/streams")
async def create_stream(
    stream_id: str,
    name: str,
    description: str,
    source: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crear stream."""
    try:
        system = get_streaming_system()
        stream = system.create_stream(
            stream_id=stream_id,
            name=name,
            description=description,
            source=source,
            metadata=metadata
        )
        return {
            "stream_id": stream.stream_id,
            "name": stream.name,
            "status": stream.status.value,
            "source": stream.source
        }
    except Exception as e:
        logger.error(f"Error creating stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams")
async def list_streams(
    status: Optional[str] = None
) -> Dict[str, Any]:
    """Listar streams."""
    try:
        system = get_streaming_system()
        stream_status = StreamStatus(status.lower()) if status else None
        streams = system.list_streams(status=stream_status)
        return {
            "streams": [
                {
                    "stream_id": s.stream_id,
                    "name": s.name,
                    "status": s.status.value,
                    "subscribers_count": len(s.subscribers)
                }
                for s in streams
            ],
            "count": len(streams)
        }
    except Exception as e:
        logger.error(f"Error listing streams: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/streams/{stream_id}/publish")
async def publish_data(
    stream_id: str,
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """Publicar datos en stream."""
    try:
        system = get_streaming_system()
        success = await system.publish_data(stream_id, data)
        
        if not success:
            raise HTTPException(status_code=404, detail="Stream not found or inactive")
        
        return {
            "stream_id": stream_id,
            "published": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/{stream_id}/statistics")
async def get_stream_statistics(stream_id: str) -> Dict[str, Any]:
    """Obtener estadísticas de stream."""
    try:
        system = get_streaming_system()
        stats = system.get_stream_statistics(stream_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting stream statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events")
async def publish_event(
    event_type: str,
    payload: Dict[str, Any],
    source: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Publicar evento."""
    try:
        event_bus = get_event_bus()
        event = await event_bus.publish(
            event_type=event_type,
            payload=payload,
            source=source,
            metadata=metadata
        )
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp
        }
    except Exception as e:
        logger.error(f"Error publishing event: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_event_history(
    event_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Obtener historial de eventos."""
    try:
        event_bus = get_event_bus()
        events = event_bus.get_event_history(event_type=event_type, limit=limit)
        return {
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "source": e.source,
                    "timestamp": e.timestamp
                }
                for e in events
            ],
            "count": len(events)
        }
    except Exception as e:
        logger.error(f"Error getting event history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/statistics")
async def get_event_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de eventos."""
    try:
        event_bus = get_event_bus()
        stats = event_bus.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting event statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






