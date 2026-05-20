"""
Queue Routes - Rutas para gestión de cola de tareas.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.services.queue_service import QueueService, TaskPriority
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


class EnqueueTaskRequest(BaseModel):
    """Request para agregar tarea a la cola."""
    task_id: str = Field(..., min_length=1, max_length=100)
    task_data: Dict[str, Any] = Field(..., description="Datos de la tarea")
    priority: str = Field(default="NORMAL", description="Prioridad: LOW, NORMAL, HIGH, URGENT, CRITICAL")
    scheduled_at: Optional[str] = Field(None, description="Fecha programada (ISO format)")
    max_retries: int = Field(default=3, ge=0, le=10)
    metadata: Optional[Dict[str, Any]] = Field(None)


def get_queue_service() -> QueueService:
    """Obtener servicio de cola."""
    try:
        return get_service("queue_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Queue service no disponible")


@router.post("/enqueue")
@handle_api_errors
async def enqueue_task(
    request: EnqueueTaskRequest,
    queue_service: QueueService = Depends(get_queue_service)
):
    """
    Agregar tarea a la cola.
    
    Args:
        request: Datos de la tarea
        
    Returns:
        Confirmación
    """
    # Validar prioridad
    try:
        priority = TaskPriority[request.priority.upper()]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Prioridad inválida: {request.priority}. Válidas: {[p.name for p in TaskPriority]}"
        )
    
    # Parsear scheduled_at si se proporciona
    scheduled_at = None
    if request.scheduled_at:
        from datetime import datetime
        try:
            scheduled_at = datetime.fromisoformat(request.scheduled_at.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Fecha inválida: {request.scheduled_at}. Use formato ISO 8601"
            )
    
    success = queue_service.enqueue(
        task_id=request.task_id,
        task_data=request.task_data,
        priority=priority,
        scheduled_at=scheduled_at,
        max_retries=request.max_retries,
        metadata=request.metadata
    )
    
    if not success:
        raise HTTPException(status_code=503, detail="Cola llena, no se pudo agregar la tarea")
    
    return {
        "message": "Tarea agregada a la cola",
        "task_id": request.task_id,
        "priority": priority.name,
        "queue_size": queue_service.get_queue_size()
    }


@router.get("/stats")
@handle_api_errors
async def get_queue_stats(
    queue_service: QueueService = Depends(get_queue_service)
):
    """
    Obtener estadísticas de la cola.
    
    Returns:
        Estadísticas
    """
    return queue_service.get_stats()


@router.post("/clear")
@handle_api_errors
async def clear_queue(
    queue_service: QueueService = Depends(get_queue_service)
):
    """
    Limpiar cola.
    
    Returns:
        Confirmación
    """
    queue_service.clear()
    return {"message": "Cola limpiada"}


@router.get("/size")
@handle_api_errors
async def get_queue_size(
    queue_service: QueueService = Depends(get_queue_service)
):
    """
    Obtener tamaño de la cola.
    
    Returns:
        Tamaño de la cola
    """
    return {
        "queue_size": queue_service.get_queue_size(),
        "processing_count": queue_service.get_processing_count()
    }



