"""Backup management endpoints"""
from fastapi import APIRouter, HTTPException, Form
from utils.backup_manager import get_backup_manager

router = APIRouter(prefix="/backup", tags=["Backup"])


@router.post("/create")
async def create_backup(
    backup_name: Optional[str] = Form(None),
    include_versions: bool = Form(True)
):
    """Create a backup"""
    backup_manager = get_backup_manager()
    backup_path = backup_manager.create_backup(backup_name, include_versions)
    
    return {
        "status": "success",
        "backup_path": backup_path,
        "backup_name": backup_name or "auto"
    }


@router.get("s")
async def list_backups():
    """List all backups"""
    backup_manager = get_backup_manager()
    backups = backup_manager.list_backups()
    
    return {
        "backups": backups,
        "total": len(backups)
    }


@router.post("/{backup_name}/restore")
async def restore_backup(backup_name: str):
    """Restore from backup"""
    backup_manager = get_backup_manager()
    result = backup_manager.restore_backup(backup_name)
    
    if result:
        return {
            "status": "success",
            "backup_name": backup_name
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to restore backup")

