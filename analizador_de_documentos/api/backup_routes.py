"""
Rutas para Backup y Recuperación
==================================

Endpoints para gestión de backups.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.backup_recovery import get_backup_system, BackupRecoverySystem

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/backups",
    tags=["Backup & Recovery"]
)


class CreateBackupRequest(BaseModel):
    """Request para crear backup"""
    source_paths: List[str] = Field(..., description="Rutas a respaldar")
    backup_id: Optional[str] = Field(None, description="ID del backup")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos")


@router.post("/")
async def create_backup(
    request: CreateBackupRequest,
    backup_system: BackupRecoverySystem = Depends(get_backup_system)
):
    """Crear backup"""
    try:
        backup = backup_system.create_backup(
            request.source_paths,
            request.backup_id,
            request.metadata
        )
        
        return {
            "status": "created",
            "backup_id": backup.backup_id,
            "timestamp": backup.timestamp,
            "size": backup.size
        }
    except Exception as e:
        logger.error(f"Error creando backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_backups(
    backup_system: BackupRecoverySystem = Depends(get_backup_system)
):
    """Listar todos los backups"""
    backups = backup_system.list_backups()
    return {"backups": backups}


@router.post("/{backup_id}/restore")
async def restore_backup(
    backup_id: str,
    target_path: Optional[str] = None,
    backup_system: BackupRecoverySystem = Depends(get_backup_system)
):
    """Restaurar backup"""
    try:
        success = backup_system.restore_backup(backup_id, target_path)
        
        if not success:
            raise HTTPException(status_code=404, detail="Backup no encontrado o fallido")
        
        return {"status": "restored", "backup_id": backup_id}
    except Exception as e:
        logger.error(f"Error restaurando backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{backup_id}")
async def delete_backup(
    backup_id: str,
    backup_system: BackupRecoverySystem = Depends(get_backup_system)
):
    """Eliminar backup"""
    success = backup_system.delete_backup(backup_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Backup no encontrado")
    
    return {"status": "deleted", "backup_id": backup_id}


@router.post("/cleanup")
async def cleanup_backups(
    keep_count: int = 10,
    backup_system: BackupRecoverySystem = Depends(get_backup_system)
):
    """Limpiar backups antiguos"""
    try:
        backup_system.cleanup_old_backups(keep_count)
        return {"status": "cleaned", "keep_count": keep_count}
    except Exception as e:
        logger.error(f"Error limpiando backups: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















