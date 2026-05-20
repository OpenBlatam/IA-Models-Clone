"""
Bulk Routes - Operaciones en lote
==================================

Endpoints para operaciones en lote (bulk operations).
"""

import logging
from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..models import TaskRequest, TaskResponse
from ..utils import get_agent, handle_route_errors, AgentDep
from ...core.oauth2 import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bulk", tags=["bulk"])


class BulkTaskRequest(BaseModel):
    """Request para crear múltiples tareas."""
    tasks: List[TaskRequest]
    parallel: bool = False  # Ejecutar en paralelo


class BulkTaskResponse(BaseModel):
    """Response de operación en lote."""
    total: int
    successful: int
    failed: int
    results: List[TaskResponse]
    errors: List[dict]


@router.post("/tasks", response_model=BulkTaskResponse)
@handle_route_errors("bulk creating tasks")
async def bulk_create_tasks(
    request: BulkTaskRequest,
    agent = AgentDep,
    current_user = Depends(get_current_active_user)
):
    """
    Crear múltiples tareas en lote.
    
    Args:
        request: Request con lista de tareas.
        agent: Instancia del agente (inyectada).
        current_user: Usuario autenticado.
    
    Returns:
        Resultado de la operación en lote.
    """
    results = []
    errors = []
    successful = 0
    failed = 0
    
    if request.parallel:
        # Ejecutar en paralelo
        import asyncio
        
        async def create_task(task_req: TaskRequest):
            try:
                task_id = await agent.add_task(task_req.command)
                return {
                    "success": True,
                    "task": TaskResponse(
                        task_id=task_id,
                        status="pending",
                        message="Task added successfully"
                    )
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        tasks = [create_task(task) for task in request.tasks]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, Exception):
                failed += 1
                errors.append({"error": str(response)})
            elif response["success"]:
                successful += 1
                results.append(response["task"])
            else:
                failed += 1
                errors.append(response)
    else:
        # Ejecutar secuencialmente
        for task_req in request.tasks:
            try:
                task_id = await agent.add_task(task_req.command)
                results.append(TaskResponse(
                    task_id=task_id,
                    status="pending",
                    message="Task added successfully"
                ))
                successful += 1
            except Exception as e:
                logger.error(f"Failed to create task: {e}")
                errors.append({
                    "command": task_req.command,
                    "error": str(e)
                })
                failed += 1
    
    return BulkTaskResponse(
        total=len(request.tasks),
        successful=successful,
        failed=failed,
        results=results,
        errors=errors
    )


@router.delete("/tasks", response_model=dict)
@handle_route_errors("bulk deleting tasks")
async def bulk_delete_tasks(
    task_ids: List[str],
    agent = AgentDep,
    current_user = Depends(get_current_active_user)
):
    """
    Eliminar múltiples tareas en lote.
    
    Args:
        task_ids: Lista de IDs de tareas.
        agent: Instancia del agente (inyectada).
        current_user: Usuario autenticado.
    
    Returns:
        Resultado de la operación.
    """
    successful = 0
    failed = 0
    
    for task_id in task_ids:
        try:
            # Implementar eliminación si está disponible
            if task_id in agent.tasks:
                del agent.tasks[task_id]
            successful += 1
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            failed += 1
    
    return {
        "total": len(task_ids),
        "successful": successful,
        "failed": failed
    }




