"""
API endpoints para backup y restore
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

from core.architecture.backup_restore import BackupManager, BackupType

router = APIRouter(prefix="/api/v1/backup", tags=["backup"])

# Instancia global del backup manager
_backup_manager: Optional[BackupManager] = None


def get_backup_manager() -> BackupManager:
    """Obtener instancia del backup manager"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager


class BackupRequest(BaseModel):
    """Request para crear backup"""
    backup_type: BackupType = BackupType.FULL
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    compress: bool = True


class RestoreRequest(BaseModel):
    """Request para restaurar backup"""
    backup_id: str
    restore_robots: bool = True
    restore_movements: bool = True


@router.post("/create")
async def create_backup(
    request: BackupRequest,
    manager: BackupManager = Depends(get_backup_manager)
):
    """Crear backup"""
    try:
        metadata = await manager.create_backup(
            backup_type=request.backup_type,
            description=request.description,
            tags=request.tags,
            compress=request.compress
        )
        return {
            "success": True,
            "backup_id": metadata.id,
            "timestamp": metadata.timestamp.isoformat(),
            "size_bytes": metadata.size_bytes,
            "entities_count": metadata.entities_count,
            "movements_count": metadata.movements_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore")
async def restore_backup(
    request: RestoreRequest,
    manager: BackupManager = Depends(get_backup_manager)
):
    """Restaurar backup"""
    try:
        result = await manager.restore_backup(
            backup_id=request.backup_id,
            restore_robots=request.restore_robots,
            restore_movements=request.restore_movements
        )
        return {
            "success": True,
            **result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_backups(
    tags: Optional[str] = None,
    manager: BackupManager = Depends(get_backup_manager)
):
    """Listar backups disponibles"""
    tag_list = tags.split(",") if tags else None
    backups = manager.list_backups(tags=tag_list)
    
    return {
        "backups": [
            {
                "id": b.id,
                "timestamp": b.timestamp.isoformat(),
                "backup_type": b.backup_type.value,
                "size_bytes": b.size_bytes,
                "entities_count": b.entities_count,
                "movements_count": b.movements_count,
                "description": b.description,
                "tags": b.tags
            }
            for b in backups
        ]
    }


@router.delete("/{backup_id}")
async def delete_backup(
    backup_id: str,
    manager: BackupManager = Depends(get_backup_manager)
):
    """Eliminar backup"""
    try:
        manager.delete_backup(backup_id)
        return {"success": True, "backup_id": backup_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



