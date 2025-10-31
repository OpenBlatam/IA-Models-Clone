"""
Backup Routes
Real, working backup and recovery endpoints for AI document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import asyncio
from backup_system import backup_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/backup", tags=["Backup & Recovery"])

@router.post("/create-backup")
async def create_backup(
    backup_type: str = Form("full"),
    include_data: bool = Form(True)
):
    """Create backup of system data"""
    try:
        result = await backup_system.create_backup(backup_type, include_data)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restore-backup")
async def restore_backup(
    backup_id: str = Form(...),
    restore_path: str = Form(".")
):
    """Restore from backup"""
    try:
        result = await backup_system.restore_backup(backup_id, restore_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list-backups")
async def list_backups():
    """List available backups"""
    try:
        backups = await backup_system.list_backups()
        return JSONResponse(content={
            "backups": backups,
            "total_backups": len(backups)
        })
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete-backup/{backup_id}")
async def delete_backup(backup_id: str):
    """Delete backup"""
    try:
        result = await backup_system.delete_backup(backup_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error deleting backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-backup/{backup_id}")
async def download_backup(backup_id: str):
    """Download backup archive"""
    try:
        backups = await backup_system.list_backups()
        backup = next((b for b in backups if b["backup_id"] == backup_id), None)
        
        if not backup:
            raise HTTPException(status_code=404, detail="Backup not found")
        
        archive_path = backup["archive_path"]
        if not os.path.exists(archive_path):
            raise HTTPException(status_code=404, detail="Backup file not found")
        
        return FileResponse(
            path=archive_path,
            filename=f"{backup_id}.zip",
            media_type="application/zip"
        )
    except Exception as e:
        logger.error(f"Error downloading backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backup-stats")
async def get_backup_stats():
    """Get backup statistics"""
    try:
        stats = backup_system.get_backup_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting backup stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backup-config")
async def get_backup_config():
    """Get backup configuration"""
    try:
        config = backup_system.get_backup_config()
        return JSONResponse(content=config)
    except Exception as e:
        logger.error(f"Error getting backup config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backup-info/{backup_id}")
async def get_backup_info(backup_id: str):
    """Get detailed backup information"""
    try:
        backups = await backup_system.list_backups()
        backup = next((b for b in backups if b["backup_id"] == backup_id), None)
        
        if not backup:
            raise HTTPException(status_code=404, detail="Backup not found")
        
        return JSONResponse(content=backup)
    except Exception as e:
        logger.error(f"Error getting backup info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-backup/{backup_id}")
async def verify_backup(backup_id: str):
    """Verify backup integrity"""
    try:
        backups = await backup_system.list_backups()
        backup = next((b for b in backups if b["backup_id"] == backup_id), None)
        
        if not backup:
            raise HTTPException(status_code=404, detail="Backup not found")
        
        archive_path = backup["archive_path"]
        if not os.path.exists(archive_path):
            raise HTTPException(status_code=404, detail="Backup file not found")
        
        # Verify archive integrity
        try:
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                # Test archive integrity
                bad_file = zipf.testzip()
                if bad_file:
                    return JSONResponse(content={
                        "backup_id": backup_id,
                        "status": "corrupted",
                        "corrupted_file": bad_file,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    return JSONResponse(content={
                        "backup_id": backup_id,
                        "status": "valid",
                        "file_count": len(zipf.namelist()),
                        "timestamp": datetime.now().isoformat()
                    })
        except zipfile.BadZipFile:
            return JSONResponse(content={
                "backup_id": backup_id,
                "status": "invalid",
                "error": "Not a valid ZIP file",
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error verifying backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-backup")
async def health_check_backup():
    """Backup system health check"""
    try:
        stats = backup_system.get_backup_stats()
        config = backup_system.get_backup_config()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Backup System",
            "version": "1.0.0",
            "features": {
                "full_backup": config["features"]["full_backup"],
                "data_backup": config["features"]["data_backup"],
                "config_backup": config["features"]["config_backup"],
                "app_backup": config["features"]["app_backup"],
                "doc_backup": config["features"]["doc_backup"],
                "install_backup": config["features"]["install_backup"],
                "automatic_cleanup": config["features"]["automatic_cleanup"],
                "backup_verification": True,
                "backup_download": True
            },
            "backup_stats": stats["stats"],
            "backup_config": {
                "backup_directory": config["backup_directory"],
                "max_backups": config["max_backups"],
                "backup_interval": config["backup_interval"]
            }
        })
    except Exception as e:
        logger.error(f"Error in backup health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))













