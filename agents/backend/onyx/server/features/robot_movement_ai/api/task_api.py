"""
Task API Endpoints
==================

Endpoints para async task manager y event sourcing.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional
import logging

from ..core.async_task_manager import get_async_task_manager
from ..core.event_sourcing import (
    get_event_store,
    EventType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.post("/submit")
async def submit_task(
    name: str,
    priority: int = 5,
    task_data: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Enviar tarea asíncrona."""
    try:
        manager = get_async_task_manager()
        
        # Función de ejemplo (en producción sería una función real)
        async def task_function(data):
            return {"processed": data}
        
        task_id = manager.submit(
            name=name,
            func=task_function,
            priority=priority,
            **task_data
        )
        
        return {
            "task_id": task_id,
            "name": name,
            "status": "pending"
        }
    except Exception as e:
        logger.error(f"Error submitting task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """Obtener tarea."""
    try:
        manager = get_async_task_manager()
        task = manager.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "priority": task.priority,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "result": task.result,
            "error": task.error
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_task_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de tareas."""
    try:
        manager = get_async_task_manager()
        stats = manager.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting task statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events")
async def append_event(
    aggregate_id: str,
    event_type: str,
    event_data: Dict[str, Any] = Body(...),
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Agregar evento al store."""
    try:
        store = get_event_store()
        event_type_enum = EventType(event_type.lower())
        event = store.append_event(
            aggregate_id=aggregate_id,
            event_type=event_type_enum,
            event_data=event_data,
            metadata=metadata
        )
        return {
            "event_id": event.event_id,
            "aggregate_id": event.aggregate_id,
            "event_type": event.event_type.value,
            "version": event.version
        }
    except Exception as e:
        logger.error(f"Error appending event: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/{aggregate_id}")
async def get_events(
    aggregate_id: str,
    from_version: int = Query(1, ge=1),
    to_version: Optional[int] = Query(None, ge=1)
) -> Dict[str, Any]:
    """Obtener eventos de agregado."""
    try:
        store = get_event_store()
        events = store.get_events(
            aggregate_id=aggregate_id,
            from_version=from_version,
            to_version=to_version
        )
        return {
            "aggregate_id": aggregate_id,
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "version": e.version,
                    "timestamp": e.timestamp
                }
                for e in events
            ],
            "count": len(events)
        }
    except Exception as e:
        logger.error(f"Error getting events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/statistics")
async def get_event_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de eventos."""
    try:
        store = get_event_store()
        stats = store.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting event statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


