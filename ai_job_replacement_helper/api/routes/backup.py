"""
Backup endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.backup import BackupService

router = APIRouter()
backup_service = BackupService()


@router.post("/create")
async def create_backup(
    data: Dict[str, Any],
    backup_name: Optional[str] = None
) -> Dict[str, Any]:
    """Crear backup"""
    try:
        backup_path = backup_service.create_backup(data, backup_name)
        return {
            "success": True,
            "backup_path": backup_path,
            "backup_name": backup_name or "auto",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore")
async def restore_backup(backup_path: str) -> Dict[str, Any]:
    """Restaurar desde backup"""
    try:
        data = backup_service.restore_backup(backup_path)
        return {
            "success": True,
            "data": data,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_backups() -> Dict[str, Any]:
    """Listar backups disponibles"""
    try:
        backups = backup_service.list_backups()
        return {
            "backups": backups,
            "total": len(backups),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{backup_name}")
async def delete_backup(backup_name: str) -> Dict[str, Any]:
    """Eliminar backup"""
    try:
        success = backup_service.delete_backup(backup_name)
        if not success:
            raise HTTPException(status_code=404, detail="Backup not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




