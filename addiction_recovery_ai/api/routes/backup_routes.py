"""
Backup routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, List

try:
    from services.backup_service import BackupService
except ImportError:
    from ...services.backup_service import BackupService

router = APIRouter()

backup = BackupService()


@router.post("/backup/create")
async def create_backup(
    user_id: str = Body(...),
    backup_type: str = Body("full"),
    include_data: Optional[List[str]] = Body(None)
):
    """Crea un backup de datos"""
    try:
        backup_result = backup.create_backup(user_id, backup_type, include_data)
        return JSONResponse(content=backup_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando backup: {str(e)}")


@router.post("/backup/restore")
async def restore_backup(
    user_id: str = Body(...),
    backup_id: str = Body(...)
):
    """Restaura datos desde backup"""
    try:
        restore_result = backup.restore_backup(user_id, backup_id)
        return JSONResponse(content=restore_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restaurando backup: {str(e)}")



