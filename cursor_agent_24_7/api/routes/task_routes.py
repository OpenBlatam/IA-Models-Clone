"""
Task Routes - Rutas para gestión de tareas
===========================================

Endpoints para crear y consultar tareas.
"""

import logging
from fastapi import APIRouter, Query

from ..models import TaskRequest, TaskResponse, TaskDetailResponse, TasksListResponse
from ..utils import get_agent, handle_route_errors, AgentDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.post("", response_model=TaskResponse)
@handle_route_errors("adding task")
async def add_task(request: TaskRequest, agent = AgentDep):
    """
    Agregar una nueva tarea.
    
    Args:
        request: Request con el comando a ejecutar.
        agent: Instancia del agente (inyectada).
    
    Returns:
        Información de la tarea creada.
    
    Raises:
        HTTPException: Si hay error al agregar la tarea.
    """
    task_id = await agent.add_task(request.command)
    return TaskResponse(
        task_id=task_id,
        status="pending",
        message="Task added successfully"
    )


@router.get("", response_model=TasksListResponse)
@handle_route_errors("getting tasks")
async def get_tasks(
    limit: int = Query(50, ge=1, le=1000, description="Número máximo de tareas"),
    agent = AgentDep
):
    """
    Obtener lista de tareas.
    
    Args:
        limit: Número máximo de tareas a retornar (1-1000).
        agent: Instancia del agente (inyectada).
    
    Returns:
        Lista de tareas.
    
    Raises:
        HTTPException: Si hay error al obtener las tareas.
    """
    tasks_data = await agent.get_tasks(limit=limit)
    
    tasks = [
        TaskDetailResponse(
            id=task["id"],
            command=task["command"],
            status=task["status"],
            timestamp=task["timestamp"],
            result=task.get("result"),
            error=task.get("error"),
        )
        for task in tasks_data
    ]
    
    return TasksListResponse(tasks=tasks, total=len(tasks))

