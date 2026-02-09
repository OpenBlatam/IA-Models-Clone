"""
Agent Routes
============

Rutas para controlar el agente.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ...core.agent import GitHubAutonomousAgent, AgentStatus
from ...core.service import PersistentService
from ..models.schemas import AgentStatusResponse, AgentControlRequest

router = APIRouter()

_agent_instance: GitHubAutonomousAgent | None = None
_service_instance: PersistentService | None = None


def get_agent() -> GitHubAutonomousAgent:
    """Obtener instancia del agente."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = GitHubAutonomousAgent()
    return _agent_instance


def get_service() -> PersistentService:
    """Obtener instancia del servicio."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PersistentService()
    return _service_instance


@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status() -> AgentStatusResponse:
    """Obtener el estado actual del agente."""
    try:
        agent = get_agent()
        status = await agent.get_status()
        
        # Asegurar que el formato coincida con lo que espera el frontend
        from datetime import datetime
        
        return AgentStatusResponse(
            status=status.get("status", "idle"),
            running_tasks=status.get("running_tasks", 0),
            queue={
                "pending": status.get("queue", {}).get("pending", 0),
                "running": status.get("queue", {}).get("running", 0),
                "completed": status.get("queue", {}).get("completed", 0),
                "failed": status.get("queue", {}).get("failed", 0)
            },
            timestamp=status.get("timestamp", datetime.now().isoformat())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_agent() -> Dict[str, Any]:
    """Iniciar el agente."""
    try:
        agent = get_agent()
        await agent.start()
        return {"message": "Agente iniciado correctamente", "status": "running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_agent() -> Dict[str, Any]:
    """Detener el agente."""
    try:
        agent = get_agent()
        await agent.stop()
        return {"message": "Agente detenido correctamente", "status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pause")
async def pause_agent() -> Dict[str, Any]:
    """Pausar el agente."""
    try:
        agent = get_agent()
        await agent.pause()
        return {"message": "Agente pausado correctamente", "status": "paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume")
async def resume_agent() -> Dict[str, Any]:
    """Reanudar el agente."""
    try:
        agent = get_agent()
        await agent.resume()
        return {"message": "Agente reanudado correctamente", "status": "running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

Rutas de la API del agente
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime

from ..models.schemas import (
    InstructionRequest,
    TaskResponse,
    AgentStatusResponse,
    StopRequest,
    TaskListResponse,
    TaskStatus
)
from ...core.executor.task_executor import TaskExecutor
from ...core.queue.task_queue import TaskQueue
from ...core.github.client import GitHubClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agent"])

task_queue = TaskQueue()
github_client = GitHubClient()
task_executor = TaskExecutor(github_client, task_queue)

executor_task = None


@router.post("/instructions", response_model=TaskResponse)
async def create_instruction(
    request: InstructionRequest,
    background_tasks: BackgroundTasks
):
    """
    Crear una nueva instrucción para el agente
    
    Args:
        request: Datos de la instrucción
        background_tasks: Tareas en background de FastAPI
        
    Returns:
        Tarea creada
    """
    try:
        task_id = task_queue.add_task({
            "github_repo": request.github_repo,
            "instruction": request.instruction,
            "priority": request.priority.value,
            "metadata": request.metadata or {}
        })
        
        task = task_queue.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=500, detail="Error al crear la tarea")
        
        if not task_executor.is_running:
            background_tasks.add_task(task_executor.run_continuous)
        
        return TaskResponse(
            id=task["id"],
            github_repo=task["github_repo"],
            instruction=task["instruction"],
            status=TaskStatus(task["status"]),
            priority=task.get("priority", "medium"),
            created_at=task["created_at"],
            started_at=task.get("started_at"),
            completed_at=task.get("completed_at"),
            result=task.get("result"),
            error=task.get("error"),
            metadata=task.get("metadata")
        )
        
    except Exception as e:
        logger.error(f"Error creando instrucción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: TaskStatus = None,
    page: int = 1,
    page_size: int = 20
):
    """
    Listar tareas
    
    Args:
        status: Filtrar por estado
        page: Página
        page_size: Tamaño de página
        
    Returns:
        Lista de tareas
    """
    try:
        status_str = status.value if status else None
        tasks = task_queue.list_tasks(status=status_str, limit=page_size * 10)
        
        start = (page - 1) * page_size
        end = start + page_size
        paginated_tasks = tasks[start:end]
        
        task_responses = [
            TaskResponse(
                id=t["id"],
                github_repo=t["github_repo"],
                instruction=t["instruction"],
                status=TaskStatus(t["status"]),
                priority=t.get("priority", "medium"),
                created_at=t["created_at"],
                started_at=t.get("started_at"),
                completed_at=t.get("completed_at"),
                result=t.get("result"),
                error=t.get("error"),
                metadata=t.get("metadata")
            )
            for t in paginated_tasks
        ]
        
        return TaskListResponse(
            tasks=task_responses,
            total=len(tasks),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listando tareas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """
    Obtener tarea por ID
    
    Args:
        task_id: ID de la tarea
        
    Returns:
        Tarea
    """
    task = task_queue.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    
    return TaskResponse(
        id=task["id"],
        github_repo=task["github_repo"],
        instruction=task["instruction"],
        status=TaskStatus(task["status"]),
        priority=task.get("priority", "medium"),
        created_at=task["created_at"],
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        result=task.get("result"),
        error=task.get("error"),
        metadata=task.get("metadata")
    )


@router.get("/status", response_model=AgentStatusResponse)
async def get_status():
    """
    Obtener estado del agente
    
    Returns:
        Estado del agente
    """
    try:
        all_tasks = task_queue.list_tasks()
        
        active = len([t for t in all_tasks if t["status"] == "running"])
        pending = len([t for t in all_tasks if t["status"] == "pending"])
        completed = len([t for t in all_tasks if t["status"] == "completed"])
        failed = len([t for t in all_tasks if t["status"] == "failed"])
        
        return AgentStatusResponse(
            is_running=task_executor.is_running,
            active_tasks=active,
            pending_tasks=pending,
            completed_tasks=completed,
            failed_tasks=failed,
            uptime_seconds=0.0
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_agent(request: StopRequest):
    """
    Detener el agente o tareas específicas
    
    Args:
        request: Request de detención
        
    Returns:
        Confirmación
    """
    try:
        if request.stop_all:
            task_executor.stop()
            return {"message": "Agente detenido", "stopped_all": True}
        else:
            if request.task_ids:
                stopped = []
                for task_id in request.task_ids:
                    task_queue.update_task(task_id, {"status": "cancelled"})
                    stopped.append(task_id)
                return {
                    "message": f"Tareas detenidas: {len(stopped)}",
                    "stopped_tasks": stopped
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Debe proporcionar task_ids o stop_all=True"
                )
                
    except Exception as e:
        logger.error(f"Error deteniendo agente: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    Eliminar tarea
    
    Args:
        task_id: ID de la tarea
        
    Returns:
        Confirmación
    """
    success = task_queue.remove_task(task_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    
    return {"message": "Tarea eliminada", "task_id": task_id}


