"""
Routines API Routes
===================

Endpoints para gestión de rutinas.
"""

import os
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from datetime import time
from pydantic import BaseModel

router = APIRouter(prefix="/routines", tags=["routines"])


class RoutineTaskCreate(BaseModel):
    title: str
    description: str
    routine_type: str
    scheduled_time: str  # HH:MM:SS format
    duration_minutes: int
    priority: int = 5
    days_of_week: List[int] = None
    is_required: bool = True
    notes: Optional[str] = None


class RoutineTaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_time: Optional[str] = None
    duration_minutes: Optional[int] = None
    priority: Optional[int] = None
    notes: Optional[str] = None


class RoutineCompletionCreate(BaseModel):
    status: str = "completed"
    notes: Optional[str] = None
    actual_duration_minutes: Optional[int] = None


def get_artist_manager(artist_id: str):
    """Dependency para obtener ArtistManager."""
    from ...core.artist_manager import ArtistManager
    from ...core.routine_manager import RoutineTask, RoutineType, RoutineStatus
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    manager = ArtistManager(artist_id=artist_id, openrouter_api_key=openrouter_key)
    return manager, RoutineTask, RoutineType, RoutineStatus


@router.post("/{artist_id}/tasks", response_model=Dict[str, Any])
async def create_routine(artist_id: str, routine: RoutineTaskCreate):
    """Crear nueva rutina."""
    try:
        manager, RoutineTask, RoutineType, _ = get_artist_manager(artist_id)
        
        import uuid
        routine_id = str(uuid.uuid4())
        
        routine_type = RoutineType(routine.routine_type) if routine.routine_type in [rt.value for rt in RoutineType] else RoutineType.CUSTOM
        
        scheduled_time = time.fromisoformat(routine.scheduled_time)
        
        routine_task = RoutineTask(
            id=routine_id,
            title=routine.title,
            description=routine.description,
            routine_type=routine_type,
            scheduled_time=scheduled_time,
            duration_minutes=routine.duration_minutes,
            priority=routine.priority,
            days_of_week=routine.days_of_week or list(range(7)),
            is_required=routine.is_required,
            notes=routine.notes
        )
        
        created_routine = manager.routines.add_routine(routine_task)
        return created_routine.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/tasks", response_model=List[Dict[str, Any]])
async def get_routines(
    artist_id: str,
    routine_type: Optional[str] = None,
    today_only: bool = False
):
    """Obtener rutinas."""
    try:
        manager, _, RoutineType, _ = get_artist_manager(artist_id)
        
        if today_only:
            routines = manager.routines.get_routines_for_today()
        elif routine_type:
            rt = RoutineType(routine_type)
            routines = manager.routines.get_routines_by_type(rt)
        else:
            routines = list(manager.routines.routines.values())
        
        return [r.to_dict() for r in routines]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/tasks/{task_id}", response_model=Dict[str, Any])
async def get_routine(artist_id: str, task_id: str):
    """Obtener rutina específica."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        routine = manager.routines.get_routine(task_id)
        if not routine:
            raise HTTPException(status_code=404, detail="Routine not found")
        return routine.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{artist_id}/tasks/{task_id}", response_model=Dict[str, Any])
async def update_routine(artist_id: str, task_id: str, routine_update: RoutineTaskUpdate):
    """Actualizar rutina."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        
        updates = routine_update.dict(exclude_unset=True)
        if "scheduled_time" in updates:
            updates["scheduled_time"] = time.fromisoformat(updates["scheduled_time"])
        
        updated_routine = manager.routines.update_routine(task_id, **updates)
        return updated_routine.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{artist_id}/tasks/{task_id}")
async def delete_routine(artist_id: str, task_id: str):
    """Eliminar rutina."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        success = manager.routines.delete_routine(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Routine not found")
        return {"status": "deleted", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artist_id}/tasks/{task_id}/complete", response_model=Dict[str, Any])
async def complete_routine(artist_id: str, task_id: str, completion: RoutineCompletionCreate):
    """Marcar rutina como completada."""
    try:
        manager, _, _, RoutineStatus = get_artist_manager(artist_id)
        
        status = RoutineStatus(completion.status) if completion.status in [s.value for s in RoutineStatus] else RoutineStatus.COMPLETED
        
        completion_record = manager.routines.mark_completed(
            task_id=task_id,
            status=status,
            notes=completion.notes,
            actual_duration_minutes=completion.actual_duration_minutes
        )
        return completion_record.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/tasks/{task_id}/completion-rate", response_model=Dict[str, Any])
async def get_completion_rate(artist_id: str, task_id: str, days: int = 30):
    """Obtener tasa de completación de rutina."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        rate = manager.routines.get_completion_rate(task_id, days=days)
        return {
            "task_id": task_id,
            "completion_rate": rate,
            "days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/pending", response_model=List[Dict[str, Any]])
async def get_pending_routines(artist_id: str):
    """Obtener rutinas pendientes para hoy."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        pending = manager.routines.get_pending_routines()
        return [r.to_dict() for r in pending]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




