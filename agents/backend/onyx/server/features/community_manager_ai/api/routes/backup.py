"""
Backup API Routes
=================

Endpoints para backup y restauración.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import FileResponse
from typing import List

router = APIRouter(prefix="/backup", tags=["backup"])


def get_community_manager():
    """Dependency para obtener CommunityManager"""
    from ...core.community_manager import CommunityManager
    return CommunityManager()


def get_backup_service():
    """Dependency para obtener BackupService"""
    from ...services.backup_service import BackupService
    return BackupService()


@router.post("/create", response_model=dict)
async def create_backup(
    include_media: bool = True,
    manager = Depends(get_community_manager),
    backup_service = Depends(get_backup_service)
):
    """Crear backup completo"""
    try:
        backup_file = backup_service.create_backup(manager, include_media)
        return {
            "status": "success",
            "backup_file": backup_file,
            "message": "Backup creado exitosamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[dict])
async def list_backups(backup_service = Depends(get_backup_service)):
    """Listar backups disponibles"""
    try:
        backups = backup_service.list_backups()
        return backups
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_backup(
    filename: str,
    backup_service = Depends(get_backup_service)
):
    """Descargar un backup"""
    try:
        from pathlib import Path
        backup_path = backup_service.backup_path / filename
        
        if not backup_path.exists():
            raise HTTPException(status_code=404, detail="Backup no encontrado")
        
        return FileResponse(
            backup_path,
            media_type="application/zip",
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore")
async def restore_backup(
    file: UploadFile = File(...),
    manager = Depends(get_community_manager),
    backup_service = Depends(get_backup_service)
):
    """Restaurar desde backup"""
    try:
        import tempfile
        import os
        
        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Restaurar
        results = backup_service.restore_backup(tmp_path, manager)
        
        # Limpiar archivo temporal
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "results": results,
            "message": "Backup restaurado exitosamente"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




