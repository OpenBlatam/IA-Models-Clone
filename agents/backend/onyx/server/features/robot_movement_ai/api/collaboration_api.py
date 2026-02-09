"""
Collaboration API Endpoints
============================

Endpoints para colaboración y auditoría.
"""

from fastapi import APIRouter, HTTPException, Query, Header
from typing import Dict, Any, Optional, List
import logging

from ..core.collaboration_system import (
    get_collaboration_system,
    TaskStatus,
    UserRole
)
from ..core.audit_log import (
    get_audit_logger,
    AuditAction,
    AuditLevel
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/collaboration", tags=["collaboration"])


@router.post("/users")
async def create_user(
    user_id: str,
    username: str,
    email: str,
    role: str = "viewer",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crear usuario."""
    try:
        system = get_collaboration_system()
        user_role = UserRole(role.lower())
        user = system.create_user(
            user_id=user_id,
            username=username,
            email=email,
            role=user_role,
            metadata=metadata
        )
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value
        }
    except Exception as e:
        logger.error(f"Error creating user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks")
async def create_task(
    task_id: str,
    title: str,
    description: str,
    created_by: str,
    assigned_to: Optional[str] = None,
    priority: int = 5,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Crear tarea."""
    try:
        system = get_collaboration_system()
        task = system.create_task(
            task_id=task_id,
            title=title,
            description=description,
            created_by=created_by,
            assigned_to=assigned_to,
            priority=priority,
            tags=tags
        )
        return {
            "task_id": task.task_id,
            "title": task.title,
            "status": task.status.value,
            "assigned_to": task.assigned_to
        }
    except Exception as e:
        logger.error(f"Error creating task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """Obtener tarea."""
    try:
        system = get_collaboration_system()
        task = system.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        comments = system.get_task_comments(task_id)
        
        return {
            "task_id": task.task_id,
            "title": task.title,
            "description": task.description,
            "status": task.status.value,
            "assigned_to": task.assigned_to,
            "created_by": task.created_by,
            "priority": task.priority,
            "tags": task.tags,
            "comments_count": len(comments),
            "created_at": task.created_at,
            "updated_at": task.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/status")
async def update_task_status(
    task_id: str,
    status: str,
    user_id: str = Header(..., alias="X-User-ID")
) -> Dict[str, Any]:
    """Actualizar estado de tarea."""
    try:
        system = get_collaboration_system()
        task_status = TaskStatus(status.lower())
        task = system.update_task_status(task_id, task_status, user_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "task_id": task.task_id,
            "status": task.status.value
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/assign")
async def assign_task(
    task_id: str,
    user_id: str,
    assigned_by: str = Header(..., alias="X-User-ID")
) -> Dict[str, Any]:
    """Asignar tarea."""
    try:
        system = get_collaboration_system()
        task = system.assign_task(task_id, user_id, assigned_by)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "task_id": task.task_id,
            "assigned_to": task.assigned_to
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/comments")
async def add_comment(
    task_id: str,
    content: str,
    user_id: str = Header(..., alias="X-User-ID")
) -> Dict[str, Any]:
    """Agregar comentario a tarea."""
    try:
        system = get_collaboration_system()
        comment = system.add_comment(task_id, user_id, content)
        
        if not comment:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "comment_id": comment.comment_id,
            "task_id": comment.task_id,
            "user_id": comment.user_id,
            "content": comment.content,
            "created_at": comment.created_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding comment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def list_tasks(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Listar tareas."""
    try:
        system = get_collaboration_system()
        
        if user_id:
            task_status = TaskStatus(status.lower()) if status else None
            tasks = system.get_user_tasks(user_id, status=task_status)
        else:
            tasks = list(system.tasks.values())[:limit]
        
        return {
            "tasks": [
                {
                    "task_id": t.task_id,
                    "title": t.title,
                    "status": t.status.value,
                    "assigned_to": t.assigned_to,
                    "priority": t.priority
                }
                for t in tasks
            ],
            "count": len(tasks)
        }
    except Exception as e:
        logger.error(f"Error listing tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/logs")
async def query_audit_logs(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    level: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Consultar logs de auditoría."""
    try:
        audit_logger = get_audit_logger()
        
        audit_action = AuditAction(action.lower()) if action else None
        audit_level = AuditLevel(level.lower()) if level else None
        
        entries = audit_logger.query(
            user_id=user_id,
            action=audit_action,
            resource_type=resource_type,
            level=audit_level,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        return {
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "timestamp": e.timestamp,
                    "user_id": e.user_id,
                    "action": e.action.value,
                    "resource_type": e.resource_type,
                    "resource_id": e.resource_id,
                    "level": e.level.value,
                    "message": e.message
                }
                for e in entries
            ],
            "count": len(entries)
        }
    except Exception as e:
        logger.error(f"Error querying audit logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/statistics")
async def get_audit_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de auditoría."""
    try:
        audit_logger = get_audit_logger()
        stats = audit_logger.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting audit statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






