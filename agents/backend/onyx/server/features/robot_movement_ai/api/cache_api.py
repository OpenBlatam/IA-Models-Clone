"""
Cache API Endpoints
==================

Endpoints para cache warming y connection pools.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from ..core.cache_warming import get_cache_warmer
from ..core.connection_pool import get_connection_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cache", tags=["cache"])


@router.post("/warming/tasks")
async def register_warming_task(
    task_id: str,
    name: str,
    priority: int = 5,
    enabled: bool = True
) -> Dict[str, Any]:
    """Registrar tarea de precalentamiento."""
    try:
        warmer = get_cache_warmer()
        
        # Función de ejemplo (en producción sería una función real)
        async def warmup_func():
            return {"status": "warmed"}
        
        task = warmer.register_task(
            task_id=task_id,
            name=name,
            warmup_func=warmup_func,
            priority=priority,
            enabled=enabled
        )
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "priority": task.priority,
            "enabled": task.enabled
        }
    except Exception as e:
        logger.error(f"Error registering warming task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/warming/tasks")
async def list_warming_tasks() -> Dict[str, Any]:
    """Listar tareas de precalentamiento."""
    try:
        warmer = get_cache_warmer()
        tasks = warmer.list_tasks()
        return {
            "tasks": [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "priority": t.priority,
                    "enabled": t.enabled,
                    "last_run": t.last_run
                }
                for t in tasks
            ],
            "count": len(tasks)
        }
    except Exception as e:
        logger.error(f"Error listing warming tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warming/tasks/{task_id}/execute")
async def execute_warming_task(task_id: str) -> Dict[str, Any]:
    """Ejecutar tarea de precalentamiento."""
    try:
        warmer = get_cache_warmer()
        result = await warmer.execute_task(task_id)
        return result
    except Exception as e:
        logger.error(f"Error executing warming task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warming/warmup-all")
async def warmup_all() -> Dict[str, Any]:
    """Ejecutar todas las tareas de precalentamiento."""
    try:
        warmer = get_cache_warmer()
        result = await warmer.warmup_all()
        return result
    except Exception as e:
        logger.error(f"Error warming up all: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/warming/history")
async def get_warming_history(
    task_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Obtener historial de precalentamiento."""
    try:
        warmer = get_cache_warmer()
        history = warmer.get_execution_history(task_id=task_id, limit=limit)
        return {
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting warming history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pools/{pool_name}/statistics")
async def get_pool_statistics(pool_name: str) -> Dict[str, Any]:
    """Obtener estadísticas de pool de conexiones."""
    try:
        pool = get_connection_pool(pool_name)
        if not pool:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        stats = pool.get_statistics()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pool statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






