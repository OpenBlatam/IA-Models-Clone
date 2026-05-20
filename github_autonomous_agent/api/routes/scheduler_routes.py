"""
Scheduler Routes - Rutas para gestión de tareas programadas.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.services import SchedulerService, ScheduleType
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


class ScheduleTaskRequest(BaseModel):
    """Request para programar tarea."""
    task_id: str = Field(..., min_length=1, max_length=100)
    schedule_type: str = Field(..., description="Tipo: once, interval, daily, weekly")
    schedule_config: Dict[str, Any] = Field(..., description="Configuración del schedule")
    task_data: Dict[str, Any] = Field(..., description="Datos de la tarea a ejecutar")
    enabled: bool = Field(default=True)
    max_runs: Optional[int] = Field(None, ge=1, description="Número máximo de ejecuciones")
    metadata: Optional[Dict[str, Any]] = Field(None)


def get_scheduler_service() -> SchedulerService:
    """Obtener servicio de scheduler."""
    try:
        return get_service("scheduler_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Scheduler service no disponible")


@router.post("/tasks")
@handle_api_errors
async def schedule_task(
    request: ScheduleTaskRequest,
    scheduler_service: SchedulerService = Depends(get_scheduler_service)
):
    """
    Programar tarea.
    
    Args:
        request: Datos de la tarea programada
        
    Returns:
        Tarea programada
    """
    # Validar schedule_type
    try:
        schedule_type = ScheduleType(request.schedule_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de schedule inválido: {request.schedule_type}"
        )
    
    # Validar schedule_config según tipo
    if schedule_type == ScheduleType.INTERVAL:
        if "interval_seconds" not in request.schedule_config:
            raise HTTPException(
                status_code=400,
                detail="schedule_config debe contener 'interval_seconds' para tipo interval"
            )
    
    elif schedule_type == ScheduleType.DAILY:
        if "hour" not in request.schedule_config or "minute" not in request.schedule_config:
            raise HTTPException(
                status_code=400,
                detail="schedule_config debe contener 'hour' y 'minute' para tipo daily"
            )
    
    elif schedule_type == ScheduleType.WEEKLY:
        if "day_of_week" not in request.schedule_config:
            raise HTTPException(
                status_code=400,
                detail="schedule_config debe contener 'day_of_week' para tipo weekly"
            )
    
    scheduled_task = scheduler_service.schedule_task(
        task_id=request.task_id,
        schedule_type=schedule_type,
        schedule_config=request.schedule_config,
        task_data=request.task_data,
        enabled=request.enabled,
        max_runs=request.max_runs,
        metadata=request.metadata
    )
    
    return {
        "task_id": scheduled_task.task_id,
        "schedule_type": scheduled_task.schedule_type.value,
        "schedule_config": scheduled_task.schedule_config,
        "enabled": scheduled_task.enabled,
        "next_run": scheduled_task.next_run.isoformat() if scheduled_task.next_run else None,
        "max_runs": scheduled_task.max_runs
    }


@router.get("/tasks")
@handle_api_errors
async def list_scheduled_tasks(
    enabled_only: bool = False,
    scheduler_service: SchedulerService = Depends(get_scheduler_service)
):
    """
    Listar tareas programadas.
    
    Args:
        enabled_only: Solo tareas habilitadas
        
    Returns:
        Lista de tareas programadas
    """
    tasks = scheduler_service.list_tasks(enabled_only=enabled_only)
    return {"total": len(tasks), "tasks": tasks}


@router.get("/tasks/{task_id}")
@handle_api_errors
async def get_scheduled_task(
    task_id: str,
    scheduler_service: SchedulerService = Depends(get_scheduler_service)
):
    """
    Obtener tarea programada.
    
    Args:
        task_id: ID de la tarea
        
    Returns:
        Tarea programada
    """
    task = scheduler_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Tarea programada no encontrada")
    
    return {
        "task_id": task.task_id,
        "schedule_type": task.schedule_type.value,
        "schedule_config": task.schedule_config,
        "enabled": task.enabled,
        "last_run": task.last_run.isoformat() if task.last_run else None,
        "next_run": task.next_run.isoformat() if task.next_run else None,
        "run_count": task.run_count,
        "max_runs": task.max_runs,
        "metadata": task.metadata
    }


@router.post("/tasks/{task_id}/enable")
@handle_api_errors
async def enable_scheduled_task(
    task_id: str,
    scheduler_service: SchedulerService = Depends(get_scheduler_service)
):
    """Habilitar tarea programada."""
    success = scheduler_service.enable_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tarea programada no encontrada")
    
    return {"message": "Tarea habilitada"}


@router.post("/tasks/{task_id}/disable")
@handle_api_errors
async def disable_scheduled_task(
    task_id: str,
    scheduler_service: SchedulerService = Depends(get_scheduler_service)
):
    """Deshabilitar tarea programada."""
    success = scheduler_service.disable_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tarea programada no encontrada")
    
    return {"message": "Tarea deshabilitada"}


@router.delete("/tasks/{task_id}")
@handle_api_errors
async def delete_scheduled_task(
    task_id: str,
    scheduler_service: SchedulerService = Depends(get_scheduler_service)
):
    """Eliminar tarea programada."""
    success = scheduler_service.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tarea programada no encontrada")
    
    return {"message": "Tarea eliminada"}


@router.get("/stats")
@handle_api_errors
async def get_scheduler_stats(
    scheduler_service: SchedulerService = Depends(get_scheduler_service)
):
    """Obtener estadísticas del scheduler."""
    return scheduler_service.get_stats()



