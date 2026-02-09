"""
Tasks API Endpoints
===================

Endpoints para gestión de tareas y flujos de trabajo.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
import logging

try:
    from ..core.scheduler import get_task_scheduler
except ImportError:
    def get_task_scheduler():
        return None
try:
    from ..core.workflow import get_workflow_manager
except ImportError:
    def get_workflow_manager():
        return None
try:
    from ..core.rate_limiter import get_rate_limiter
except ImportError:
    def get_rate_limiter():
        return None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.get("/scheduled")
async def list_scheduled_tasks() -> Dict[str, Any]:
    """Listar tareas programadas."""
    try:
        scheduler = get_task_scheduler()
        tasks = scheduler.list_tasks()
        return {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "schedule_type": task.schedule_type,
                    "status": task.status.value,
                    "enabled": task.enabled,
                    "run_count": task.run_count,
                    "error_count": task.error_count,
                    "last_run": task.last_run,
                    "next_run": task.next_run
                }
                for task in tasks
            ],
            "count": len(tasks)
        }
    except Exception as e:
        logger.error(f"Error listing tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduled")
async def create_scheduled_task(
    task_id: str,
    name: str,
    schedule_type: str = "interval",
    schedule_value: Any = 60,
    enabled: bool = True
) -> Dict[str, Any]:
    """Crear tarea programada."""
    try:
        scheduler = get_task_scheduler()
        # Nota: En producción, recibir la función como parámetro o referencia
        # Por ahora, solo creamos la estructura
        return {
            "message": "Task creation endpoint - implement function reference",
            "task_id": task_id,
            "name": name
        }
    except Exception as e:
        logger.error(f"Error creating task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduled/{task_id}/enable")
async def enable_task(task_id: str) -> Dict[str, Any]:
    """Habilitar tarea."""
    try:
        scheduler = get_task_scheduler()
        if scheduler.enable_task(task_id):
            return {"message": f"Task {task_id} enabled"}
        raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
        logger.error(f"Error enabling task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduled/{task_id}/disable")
async def disable_task(task_id: str) -> Dict[str, Any]:
    """Deshabilitar tarea."""
    try:
        scheduler = get_task_scheduler()
        if scheduler.disable_task(task_id):
            return {"message": f"Task {task_id} disabled"}
        raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
        logger.error(f"Error disabling task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows")
async def list_workflows() -> Dict[str, Any]:
    """Listar flujos de trabajo."""
    try:
        manager = get_workflow_manager()
        workflows = manager.list_workflows()
        return {
            "workflows": [
                {
                    "workflow_id": w.workflow_id,
                    "name": w.name,
                    "steps_count": len(w.steps),
                    "executions_count": len(w.execution_history)
                }
                for w in workflows
            ],
            "count": len(workflows)
        }
    except Exception as e:
        logger.error(f"Error listing workflows: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Ejecutar flujo de trabajo."""
    try:
        manager = get_workflow_manager()
        workflow = manager.get_workflow(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        result = await workflow.execute(context=context)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rate-limits")
async def list_rate_limits() -> Dict[str, Any]:
    """Listar límites de tasa."""
    try:
        limiter = get_rate_limiter()
        limits_info = {}
        
        for key in limiter.limits.keys():
            info = limiter.get_limit_info(key)
            if info:
                limits_info[key] = info
        
        return {
            "limits": limits_info,
            "count": len(limits_info)
        }
    except Exception as e:
        logger.error(f"Error listing rate limits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rate-limits/{key}/reset")
async def reset_rate_limit(key: str) -> Dict[str, Any]:
    """Resetear límite de tasa."""
    try:
        limiter = get_rate_limiter()
        limiter.reset_limit(key)
        return {"message": f"Rate limit reset for {key}"}
    except Exception as e:
        logger.error(f"Error resetting rate limit: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






