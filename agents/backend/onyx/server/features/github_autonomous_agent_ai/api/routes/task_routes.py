"""
Task Routes
===========

Rutas para gestionar tareas.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional

from ...core.agent import GitHubAutonomousAgent
from ..models.schemas import TaskCreateRequest, TaskResponse, TaskListResponse

router = APIRouter()

_agent_instance: GitHubAutonomousAgent | None = None


def get_agent() -> GitHubAutonomousAgent:
    """Obtener instancia del agente."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = GitHubAutonomousAgent()
    return _agent_instance


@router.post("/", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest) -> TaskResponse:
    """Crear una nueva tarea."""
    try:
        agent = get_agent()
        task_id = await agent.add_task(
            repository=request.repository,
            instruction=request.instruction,
            metadata=request.metadata or {}
        )
        
        return TaskResponse(
            id=task_id,
            repository=request.repository,
            instruction=request.instruction,
            status="pending",
            metadata=request.metadata or {}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = None,
    repository: Optional[str] = None
) -> TaskListResponse:
    """Listar tareas."""
    try:
        from ...core.task_queue import TaskQueue
        
        queue = TaskQueue()
        await queue.initialize()
        
        queue_status = await queue.get_status()
        
        return TaskListResponse(
            tasks=[],
            status=queue_status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """Obtener una tarea específica."""
    try:
        import aiosqlite
        from ...config.settings import settings
        
        async with aiosqlite.connect(settings.TASKS_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    raise HTTPException(status_code=404, detail="Tarea no encontrada")
                    
                task = dict(row)
                import json
                task["metadata"] = json.loads(task["metadata"] or "{}")
                task["result"] = json.loads(task["result"] or "{}")
                
                return TaskResponse(**task)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




