"""
Stats Routes - Rutas para estadísticas y métricas.
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
from datetime import datetime

from api.utils import handle_api_errors
from api.dependencies import get_storage
from config.di_setup import get_service
from core.storage import TaskStorage
from config.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/overview")
@handle_api_errors
async def get_stats_overview(storage: TaskStorage = Depends(get_storage)):
    """
    Obtener resumen de estadísticas.
    
    Returns:
        Diccionario con estadísticas generales
    """
    try:
        # Obtener todas las tareas
        all_tasks = await storage.get_tasks(status=None, limit=10000)
        
        # Calcular estadísticas
        total_tasks = len(all_tasks)
        tasks_by_status = {}
        tasks_by_repository = {}
        
        for task in all_tasks:
            # Por estado
            status = task.get("status", "unknown")
            tasks_by_status[status] = tasks_by_status.get(status, 0) + 1
            
            # Por repositorio
            repo_key = f"{task.get('repository_owner')}/{task.get('repository_name')}"
            tasks_by_repository[repo_key] = tasks_by_repository.get(repo_key, 0) + 1
        
        # Obtener métricas de servicios
        cache_stats = {}
        metrics_data = {}
        
        try:
            cache_service = get_service("cache_service")
            cache_stats = cache_service.get_stats()
        except Exception:
            pass
        
        try:
            metrics_service = get_service("metrics_service")
            metrics_data = metrics_service.get_metrics()
        except Exception:
            pass
        
        # Obtener estado del agente
        agent_state = await storage.get_agent_state()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "tasks": {
                "total": total_tasks,
                "by_status": tasks_by_status,
                "by_repository": dict(sorted(
                    tasks_by_repository.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])  # Top 10 repositorios
            },
            "agent": {
                "is_running": agent_state.get("is_running", False),
                "current_task_id": agent_state.get("current_task_id")
            },
            "cache": cache_stats,
            "metrics": metrics_data
        }
    except Exception as e:
        logger.error(f"Error getting stats overview: {e}", exc_info=True)
        raise


@router.get("/tasks/summary")
@handle_api_errors
async def get_tasks_summary(storage: TaskStorage = Depends(get_storage)):
    """
    Obtener resumen de tareas.
    
    Returns:
        Resumen de tareas por estado y repositorio
    """
    try:
        all_tasks = await storage.get_tasks(status=None, limit=10000)
        
        summary = {
            "total": len(all_tasks),
            "by_status": {},
            "by_repository": {},
            "recent_tasks": []
        }
        
        # Agrupar por estado
        for task in all_tasks:
            status = task.get("status", "unknown")
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            
            repo_key = f"{task.get('repository_owner')}/{task.get('repository_name')}"
            if repo_key not in summary["by_repository"]:
                summary["by_repository"][repo_key] = {
                    "total": 0,
                    "by_status": {}
                }
            summary["by_repository"][repo_key]["total"] += 1
            summary["by_repository"][repo_key]["by_status"][status] = \
                summary["by_repository"][repo_key]["by_status"].get(status, 0) + 1
        
        # Tareas recientes (últimas 10)
        sorted_tasks = sorted(
            all_tasks,
            key=lambda t: t.get("updated_at", ""),
            reverse=True
        )
        summary["recent_tasks"] = [
            {
                "id": t.get("id"),
                "repository": f"{t.get('repository_owner')}/{t.get('repository_name')}",
                "status": t.get("status"),
                "updated_at": t.get("updated_at")
            }
            for t in sorted_tasks[:10]
        ]
        
        return summary
    except Exception as e:
        logger.error(f"Error getting tasks summary: {e}", exc_info=True)
        raise


@router.get("/performance")
@handle_api_errors
async def get_performance_stats():
    """
    Obtener estadísticas de rendimiento.
    
    Returns:
        Estadísticas de rendimiento de servicios
    """
    stats = {
        "timestamp": datetime.now().isoformat(),
        "cache": {},
        "metrics": {},
        "rate_limit": {}
    }
    
    try:
        cache_service = get_service("cache_service")
        stats["cache"] = cache_service.get_stats()
    except Exception:
        pass
    
    try:
        metrics_service = get_service("metrics_service")
        stats["metrics"] = metrics_service.get_metrics()
    except Exception:
        pass
    
    try:
        rate_limit_service = get_service("rate_limit_service")
        # Obtener stats para un identificador de ejemplo
        stats["rate_limit"] = rate_limit_service.get_stats("default")
    except Exception:
        pass
    
    return stats



