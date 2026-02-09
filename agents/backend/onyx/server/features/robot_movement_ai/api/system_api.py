"""
System API Endpoints
====================

Endpoints para gestión del sistema.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

try:
    from ..core.versioning.versioning import get_version_manager, get_current_version
except ImportError:
    def get_version_manager():
        return None
    def get_current_version():
        return "1.0.0"
try:
    from ..core.backup.backup import get_backup_manager
except ImportError:
    def get_backup_manager():
        return None
try:
    from ..core.dynamic_config import get_dynamic_config_manager
except ImportError:
    def get_dynamic_config_manager():
        return None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/version")
async def get_version() -> Dict[str, Any]:
    """Obtener información de versión."""
    try:
        manager = get_version_manager()
        version_info = manager.get_version_info()
        return version_info.to_dict()
    except Exception as e:
        logger.error(f"Error getting version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backups")
async def list_backups() -> Dict[str, Any]:
    """Listar backups disponibles."""
    try:
        manager = get_backup_manager()
        backups = manager.list_backups()
        return {
            "backups": backups,
            "count": len(backups)
        }
    except Exception as e:
        logger.error(f"Error listing backups: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backups/create")
async def create_backup(
    name: str = None,
    include_config: bool = True,
    include_cache: bool = False,
    include_logs: bool = False,
    include_data: bool = True
) -> Dict[str, Any]:
    """Crear backup del sistema."""
    try:
        manager = get_backup_manager()
        backup_info = manager.create_backup(
            name=name,
            include_config=include_config,
            include_cache=include_cache,
            include_logs=include_logs,
            include_data=include_data
        )
        return backup_info
    except Exception as e:
        logger.error(f"Error creating backup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backups/{backup_name}/restore")
async def restore_backup(
    backup_name: str,
    restore_config: bool = True,
    restore_cache: bool = False,
    restore_logs: bool = False,
    restore_data: bool = True
) -> Dict[str, Any]:
    """Restaurar backup."""
    try:
        manager = get_backup_manager()
        restore_info = manager.restore_backup(
            backup_name=backup_name,
            restore_config=restore_config,
            restore_cache=restore_cache,
            restore_logs=restore_logs,
            restore_data=restore_data
        )
        return restore_info
    except Exception as e:
        logger.error(f"Error restoring backup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """Obtener configuración dinámica."""
    try:
        manager = get_dynamic_config_manager()
        return manager.config
    except Exception as e:
        logger.error(f"Error getting config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/{key:path}")
async def set_config(key: str, value: Any) -> Dict[str, Any]:
    """Establecer valor de configuración."""
    try:
        manager = get_dynamic_config_manager()
        manager.set(key, value, save=True)
        return {
            "key": key,
            "value": value,
            "message": "Configuration updated"
        }
    except Exception as e:
        logger.error(f"Error setting config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/history")
async def get_config_history(limit: int = 100) -> Dict[str, Any]:
    """Obtener historial de cambios de configuración."""
    try:
        manager = get_dynamic_config_manager()
        history = manager.get_change_history(limit=limit)
        return {
            "changes": [
                {
                    "key": change.key,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "timestamp": change.timestamp
                }
                for change in history
            ],
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting config history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






