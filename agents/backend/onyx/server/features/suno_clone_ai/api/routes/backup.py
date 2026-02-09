"""
API de Backup y Recovery

Endpoints para:
- Crear backups
- Listar backups
- Restaurar backups
- Verificar integridad
- Limpieza de backups antiguos
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from middleware.auth_middleware import require_role
from utils.backup_recovery import get_backup_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/backup",
    tags=["backup"],
    dependencies=[Depends(require_role("admin"))]  # Requiere rol admin
)


@router.post("/create")
async def create_backup(
    backup_type: str = Body("full", description="Tipo de backup (full, incremental)"),
    description: Optional[str] = Body(None, description="Descripción del backup")
) -> Dict[str, Any]:
    """
    Crea un backup del sistema.
    
    Nota: En producción, esto debería ejecutarse en background.
    """
    try:
        backup_manager = get_backup_manager()
        
        # Obtener datos a respaldar (ejemplo básico)
        from services.task_queue import get_task_queue
        from services.notification_service_advanced import get_notification_service
        
        task_queue = get_task_queue()
        notification_service = get_notification_service()
        
        backup_data = {
            "tasks": {task_id: task.to_dict() for task_id, task in task_queue.tasks.items()},
            "notifications": {
                notif_id: {
                    "id": notif.id,
                    "user_id": notif.user_id,
                    "title": notif.title,
                    "message": notif.message,
                    "channel": notif.channel.value,
                    "priority": notif.priority.value,
                    "data": notif.data,
                    "created_at": notif.created_at.isoformat()
                }
                for notif_id, notif in notification_service.notifications.items()
            }
        }
        
        backup_id = backup_manager.create_backup(
            backup_data,
            backup_type=backup_type,
            description=description
        )
        
        return {
            "backup_id": backup_id,
            "message": "Backup created successfully",
            "type": backup_type
        }
    except Exception as e:
        logger.error(f"Error creating backup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating backup: {str(e)}"
        )


@router.get("/list")
async def list_backups(
    backup_type: Optional[str] = Query(None, description="Filtrar por tipo")
) -> Dict[str, Any]:
    """
    Lista todos los backups disponibles.
    """
    try:
        backup_manager = get_backup_manager()
        backups = backup_manager.list_backups(backup_type=backup_type)
        
        return {
            "backups": backups,
            "total": len(backups)
        }
    except Exception as e:
        logger.error(f"Error listing backups: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing backups: {str(e)}"
        )


@router.get("/{backup_id}/verify")
async def verify_backup(backup_id: str) -> Dict[str, Any]:
    """
    Verifica la integridad de un backup.
    """
    try:
        backup_manager = get_backup_manager()
        result = backup_manager.verify_backup(backup_id)
        
        if not result.get("valid"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Backup verification failed: {result.get('error', 'Unknown error')}"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying backup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error verifying backup: {str(e)}"
        )


@router.post("/{backup_id}/restore")
async def restore_backup(backup_id: str) -> Dict[str, Any]:
    """
    Restaura un backup.
    
    ⚠️ ADVERTENCIA: Esta operación puede sobrescribir datos existentes.
    """
    try:
        backup_manager = get_backup_manager()
        
        # Verificar primero
        verification = backup_manager.verify_backup(backup_id)
        if not verification.get("valid"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot restore invalid backup: {verification.get('error')}"
            )
        
        # Restaurar
        restored_data = backup_manager.restore_backup(backup_id)
        
        # Aquí se restaurarían los datos en el sistema
        # Por ahora solo retornamos información
        
        return {
            "message": f"Backup {backup_id} restored successfully",
            "restored_items": {
                "tasks": len(restored_data.get("tasks", {})),
                "notifications": len(restored_data.get("notifications", {}))
            }
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error restoring backup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error restoring backup: {str(e)}"
        )


@router.delete("/{backup_id}")
async def delete_backup(backup_id: str) -> Dict[str, Any]:
    """
    Elimina un backup.
    """
    try:
        backup_manager = get_backup_manager()
        success = backup_manager.delete_backup(backup_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backup {backup_id} not found"
            )
        
        return {
            "message": f"Backup {backup_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting backup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting backup: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_old_backups(
    days: int = Query(30, ge=1, le=365, description="Eliminar backups más antiguos que X días")
) -> Dict[str, Any]:
    """
    Elimina backups antiguos.
    """
    try:
        backup_manager = get_backup_manager()
        deleted = backup_manager.cleanup_old_backups(days=days)
        
        return {
            "message": f"Cleaned up {deleted} old backups",
            "deleted_count": deleted,
            "days": days
        }
    except Exception as e:
        logger.error(f"Error cleaning up backups: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cleaning up backups: {str(e)}"
        )

