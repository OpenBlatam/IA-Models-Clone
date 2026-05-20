"""
Backup Routes
=============
Endpoints para backup y restore.
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Any, Dict, Optional

from ...utils.backup_restore import BackupManager
from ...utils.logger import get_logger

router = APIRouter(prefix="/api/v1/backup", tags=["Backup & Restore"])
logger = get_logger(__name__)

# Instancia global del backup manager
backup_manager = BackupManager()


@router.post("/create")
async def create_backup(
    data: Any = Body(...),
    name: str = Body("backup"),
    compress: bool = Body(True),
    format: str = Body("json")
):
    """
    Crear backup de datos.
    
    Args:
        data: Datos a respaldar
        name: Nombre del backup
        compress: Comprimir backup
        format: Formato (json, pickle)
        
    Returns:
        Información del backup creado
    """
    try:
        backup_path = backup_manager.create_backup(data, name, compress, format)
        return {
            "status": "success",
            "message": "Backup created successfully",
            "path": backup_path
        }
    except Exception as e:
        logger.error(f"Backup creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore")
async def restore_backup(backup_path: str = Body(...)):
    """
    Restaurar backup.
    
    Args:
        backup_path: Ruta del backup
        
    Returns:
        Datos restaurados
    """
    try:
        data = backup_manager.restore_backup(backup_path)
        return {
            "status": "success",
            "message": "Backup restored successfully",
            "data": data
        }
    except Exception as e:
        logger.error(f"Backup restore error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_backups(name_pattern: Optional[str] = None):
    """
    Listar backups disponibles.
    
    Args:
        name_pattern: Patrón de nombre (opcional)
        
    Returns:
        Lista de backups
    """
    try:
        backups = backup_manager.list_backups(name_pattern)
        return {
            "status": "success",
            "backups": backups,
            "count": len(backups)
        }
    except Exception as e:
        logger.error(f"Backup list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_backups(
    days: int = Body(30),
    keep_latest: int = Body(10)
):
    """
    Limpiar backups antiguos.
    
    Args:
        days: Días de antigüedad para eliminar
        keep_latest: Mantener N más recientes
        
    Returns:
        Resultado de limpieza
    """
    try:
        backup_manager.cleanup_old_backups(days, keep_latest)
        return {
            "status": "success",
            "message": "Backup cleanup completed"
        }
    except Exception as e:
        logger.error(f"Backup cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



