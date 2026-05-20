"""
Batch Routes - Rutas para operaciones en lote.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from api.utils import handle_api_errors, validate_github_token
from api.dependencies import get_storage
from api.schemas import CreateTaskRequest
from core.storage import TaskStorage
from core.constants import SuccessMessages
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


class BatchCreateTasksRequest(BaseModel):
    """Request para crear múltiples tareas."""
    tasks: List[CreateTaskRequest] = Field(..., min_length=1, max_length=100)


class BatchTaskResponse(BaseModel):
    """Respuesta de operación batch."""
    total: int
    successful: int
    failed: int
    tasks: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]


@router.post("/tasks", response_model=BatchTaskResponse)
@handle_api_errors
async def batch_create_tasks(
    request: BatchCreateTasksRequest,
    storage: TaskStorage = Depends(get_storage)
):
    """
    Crear múltiples tareas en lote.
    
    Args:
        request: Lista de tareas a crear
        
    Returns:
        BatchTaskResponse: Resultado de la operación batch
    """
    validate_github_token()
    
    from application.use_cases.task_use_cases import CreateTaskUseCase
    from api.validators import validate_repository, validate_instruction
    
    create_use_case = CreateTaskUseCase(storage=storage)
    
    results = []
    errors = []
    successful = 0
    failed = 0
    
    for task_request in request.tasks:
        try:
            # Validar
            validated_owner, validated_repo = validate_repository(
                task_request.repository_owner,
                task_request.repository_name
            )
            validated_instruction = validate_instruction(task_request.instruction)
            
            # Crear tarea
            task = await create_use_case.execute(
                repository_owner=validated_owner,
                repository_name=validated_repo,
                instruction=validated_instruction,
                metadata=task_request.metadata
            )
            
            results.append(task)
            successful += 1
            
            # Publicar evento
            try:
                from core.events import publish_task_event, EventType
                await publish_task_event(EventType.TASK_CREATED, task)
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error creating task in batch: {e}", exc_info=True)
            errors.append({
                "repository": f"{task_request.repository_owner}/{task_request.repository_name}",
                "instruction": task_request.instruction[:50],
                "error": str(e)
            })
            failed += 1
    
    logger.info(f"Batch create: {successful} successful, {failed} failed")
    
    return BatchTaskResponse(
        total=len(request.tasks),
        successful=successful,
        failed=failed,
        tasks=results,
        errors=errors
    )


@router.delete("/tasks")
@handle_api_errors
async def batch_delete_tasks(
    task_ids: List[str] = Field(..., min_length=1, max_length=100),
    storage: TaskStorage = Depends(get_storage)
):
    """
    Eliminar múltiples tareas en lote.
    
    Args:
        task_ids: Lista de IDs de tareas a eliminar
        
    Returns:
        Diccionario con resultados
    """
    from api.validators import validate_task_id
    
    results = {
        "total": len(task_ids),
        "deleted": 0,
        "not_found": 0,
        "errors": []
    }
    
    for task_id in task_ids:
        try:
            validated_id = validate_task_id(task_id)
            task = await storage.get_task(validated_id)
            
            if not task:
                results["not_found"] += 1
                continue
            
            deleted = await storage.delete_task(validated_id)
            if deleted:
                results["deleted"] += 1
            else:
                results["errors"].append({
                    "task_id": task_id,
                    "error": "Failed to delete"
                })
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}")
            results["errors"].append({
                "task_id": task_id,
                "error": str(e)
            })
    
    logger.info(f"Batch delete: {results['deleted']} deleted, {results['not_found']} not found")
    return results


@router.post("/tasks/{status}")
@handle_api_errors
async def batch_update_task_status(
    status: str,
    task_ids: List[str] = Field(..., min_length=1, max_length=100),
    storage: TaskStorage = Depends(get_storage)
):
    """
    Actualizar estado de múltiples tareas.
    
    Args:
        status: Nuevo estado
        task_ids: Lista de IDs de tareas
        
    Returns:
        Diccionario con resultados
    """
    from core.constants import TaskStatus
    
    if status not in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail=f"Estado inválido: {status}")
    
    results = {
        "total": len(task_ids),
        "updated": 0,
        "not_found": 0,
        "errors": []
    }
    
    for task_id in task_ids:
        try:
            task = await storage.get_task(task_id)
            if not task:
                results["not_found"] += 1
                continue
            
            await storage.update_task_status(task_id, status)
            results["updated"] += 1
            
            # Broadcast update
            try:
                from api.routes.websocket_routes import broadcast_task_update
                updated_task = await storage.get_task(task_id)
                if updated_task:
                    await broadcast_task_update(updated_task)
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}")
            results["errors"].append({
                "task_id": task_id,
                "error": str(e)
            })
    
    logger.info(f"Batch update status: {results['updated']} updated")
    return results



