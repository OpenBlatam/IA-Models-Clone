"""
API de Administración

Endpoints para administración del sistema:
- Gestión de tareas
- Estadísticas del sistema
- Configuración
- Limpieza de caché
- Operaciones de mantenimiento
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from datetime import datetime, timedelta

from middleware.auth_middleware import require_role
from services.task_queue import get_task_queue, TaskStatus, TaskPriority
from services.notification_service_advanced import get_notification_service
from utils.distributed_cache import get_distributed_cache
from utils.alerting import get_alert_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_role("admin"))]  # Requiere rol admin
)


@router.get("/stats")
async def get_admin_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas completas del sistema para administración.
    """
    try:
        task_queue = get_task_queue()
        notification_service = get_notification_service()
        cache = get_distributed_cache()
        alert_manager = get_alert_manager()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "task_queue": task_queue.get_queue_stats(),
            "notifications": notification_service.get_stats(),
            "cache": cache.get_stats(),
            "alerts": alert_manager.get_stats()
        }
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )


@router.get("/tasks")
async def list_tasks(
    status_filter: Optional[str] = Query(None, description="Filtrar por estado"),
    limit: int = Query(100, ge=1, le=1000, description="Límite de resultados")
) -> Dict[str, Any]:
    """
    Lista tareas del sistema.
    """
    try:
        task_queue = get_task_queue()
        
        if status_filter:
            try:
                task_status = TaskStatus(status_filter)
                tasks = task_queue.get_tasks_by_status(task_status, limit=limit)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status_filter}"
                )
        else:
            tasks = list(task_queue.tasks.values())[:limit]
        
        return {
            "tasks": [task.to_dict() for task in tasks],
            "total": len(tasks)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing tasks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing tasks: {str(e)}"
        )


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """
    Obtiene información de una tarea específica.
    """
    try:
        task_queue = get_task_queue()
        task = task_queue.get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return task.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting task: {str(e)}"
        )


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Cancela una tarea.
    """
    try:
        task_queue = get_task_queue()
        success = task_queue.cancel_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not cancel task {task_id}"
            )
        
        return {
            "message": f"Task {task_id} cancelled",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cancelling task: {str(e)}"
        )


@router.post("/cache/clear")
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Patrón de claves a eliminar (ej: 'user:*')")
) -> Dict[str, Any]:
    """
    Limpia el caché.
    """
    try:
        cache = get_distributed_cache()
        
        if pattern:
            deleted = cache.clear_pattern(pattern)
            message = f"Cleared {deleted} keys matching pattern '{pattern}'"
        else:
            # Limpiar todo (implementación básica)
            deleted = cache.clear_pattern("*")
            message = f"Cleared all cache ({deleted} keys)"
        
        return {
            "message": message,
            "deleted_keys": deleted,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )


@router.get("/notifications")
async def list_notifications(
    user_id: Optional[str] = Query(None, description="Filtrar por usuario"),
    unread_only: bool = Query(False, description="Solo no leídas"),
    limit: int = Query(50, ge=1, le=500, description="Límite de resultados")
) -> Dict[str, Any]:
    """
    Lista notificaciones del sistema.
    """
    try:
        notification_service = get_notification_service()
        
        if user_id:
            notifications = notification_service.get_user_notifications(
                user_id, unread_only=unread_only, limit=limit
            )
        else:
            # Obtener todas (implementación básica)
            notifications = list(notification_service.notifications.values())[:limit]
        
        return {
            "notifications": [
                {
                    "id": n.id,
                    "user_id": n.user_id,
                    "title": n.title,
                    "message": n.message,
                    "channel": n.channel.value,
                    "priority": n.priority.value,
                    "delivered": n.delivered,
                    "read_at": n.read_at.isoformat() if n.read_at else None,
                    "created_at": n.created_at.isoformat()
                }
                for n in notifications
            ],
            "total": len(notifications)
        }
    except Exception as e:
        logger.error(f"Error listing notifications: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing notifications: {str(e)}"
        )


@router.post("/maintenance/cleanup")
async def run_cleanup(
    days: int = Query(30, ge=1, le=365, description="Eliminar datos más antiguos que X días")
) -> Dict[str, Any]:
    """
    Ejecuta tareas de limpieza del sistema.
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned = {
            "tasks": 0,
            "notifications": 0,
            "cache_keys": 0
        }
        
        # Limpiar tareas antiguas completadas
        task_queue = get_task_queue()
        for task_id, task in list(task_queue.tasks.items()):
            if task.status == TaskStatus.COMPLETED and task.completed_at:
                if task.completed_at < cutoff_date:
                    del task_queue.tasks[task_id]
                    cleaned["tasks"] += 1
        
        # Limpiar notificaciones antiguas leídas
        notification_service = get_notification_service()
        for notif_id, notif in list(notification_service.notifications.items()):
            if notif.read_at and notif.read_at < cutoff_date:
                del notification_service.notifications[notif_id]
                cleaned["notifications"] += 1
        
        return {
            "message": "Cleanup completed",
            "cleaned": cleaned,
            "cutoff_date": cutoff_date.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running cleanup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running cleanup: {str(e)}"
        )


@router.get("/alerts")
async def get_admin_alerts(
    hours: int = Query(24, ge=1, le=168, description="Horas de historial")
) -> Dict[str, Any]:
    """
    Obtiene alertas del sistema para administración.
    """
    try:
        alert_manager = get_alert_manager()
        history = alert_manager.get_alert_history(hours=hours)
        
        return {
            "active_alerts": [alert.to_dict() for alert in alert_manager.get_active_alerts()],
            "history": [alert.to_dict() for alert in history],
            "stats": alert_manager.get_stats(),
            "time_range_hours": hours
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving alerts: {str(e)}"
        )

