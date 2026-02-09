"""
Version Control API Endpoints
==============================

Endpoints para control de versiones y snapshots.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from ..core.version_control import get_version_control
from ..core.snapshot_manager import get_snapshot_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/version", tags=["version"])


@router.post("/entities/{entity_type}/{entity_id}/versions")
async def create_version(
    entity_type: str,
    entity_id: str,
    data: Dict[str, Any],
    created_by: Optional[str] = None,
    message: str = "",
    parent_version: Optional[str] = None
) -> Dict[str, Any]:
    """Crear versión de entidad."""
    try:
        vc = get_version_control()
        version = vc.create_version(
            entity_type=entity_type,
            entity_id=entity_id,
            data=data,
            created_by=created_by,
            message=message,
            parent_version=parent_version
        )
        return {
            "version_id": version.version_id,
            "entity_type": version.entity_type,
            "entity_id": version.entity_id,
            "timestamp": version.timestamp,
            "message": version.message
        }
    except Exception as e:
        logger.error(f"Error creating version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_type}/{entity_id}/versions")
async def get_versions(
    entity_type: str,
    entity_id: str,
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Obtener versiones de entidad."""
    try:
        vc = get_version_control()
        versions = vc.get_versions(entity_type, entity_id, limit=limit)
        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "versions": [
                {
                    "version_id": v.version_id,
                    "timestamp": v.timestamp,
                    "created_by": v.created_by,
                    "message": v.message
                }
                for v in versions
            ],
            "count": len(versions)
        }
    except Exception as e:
        logger.error(f"Error getting versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_type}/{entity_id}/versions/{version_id}")
async def get_version(
    entity_type: str,
    entity_id: str,
    version_id: str
) -> Dict[str, Any]:
    """Obtener versión específica."""
    try:
        vc = get_version_control()
        version = vc.get_version(entity_type, entity_id, version_id)
        
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return {
            "version_id": version.version_id,
            "entity_type": version.entity_type,
            "entity_id": version.entity_id,
            "data": version.data,
            "created_by": version.created_by,
            "message": version.message,
            "timestamp": version.timestamp,
            "parent_version": version.parent_version
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_type}/{entity_id}/versions/{version_id1}/diff/{version_id2}")
async def diff_versions(
    entity_type: str,
    entity_id: str,
    version_id1: str,
    version_id2: str
) -> Dict[str, Any]:
    """Comparar dos versiones."""
    try:
        vc = get_version_control()
        diff = vc.diff_versions(entity_type, entity_id, version_id1, version_id2)
        return diff
    except Exception as e:
        logger.error(f"Error diffing versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/{entity_type}/{entity_id}/versions/{version_id}/restore")
async def restore_version(
    entity_type: str,
    entity_id: str,
    version_id: str,
    created_by: Optional[str] = None
) -> Dict[str, Any]:
    """Restaurar versión."""
    try:
        vc = get_version_control()
        restored = vc.restore_version(
            entity_type=entity_type,
            entity_id=entity_id,
            version_id=version_id,
            created_by=created_by
        )
        
        if not restored:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return {
            "version_id": restored.version_id,
            "restored_from": version_id,
            "timestamp": restored.timestamp
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/snapshots")
async def create_snapshot(
    snapshot_id: str,
    name: str,
    description: str = "",
    created_by: Optional[str] = None,
    system_snapshot: bool = False
) -> Dict[str, Any]:
    """Crear snapshot."""
    try:
        manager = get_snapshot_manager()
        
        if system_snapshot:
            snapshot = manager.create_system_snapshot(
                snapshot_id=snapshot_id,
                name=name,
                description=description,
                created_by=created_by
            )
        else:
            # Snapshot manual con datos vacíos
            snapshot = manager.create_snapshot(
                snapshot_id=snapshot_id,
                name=name,
                description=description,
                data={},
                created_by=created_by
            )
        
        return {
            "snapshot_id": snapshot.snapshot_id,
            "name": snapshot.name,
            "timestamp": snapshot.timestamp
        }
    except Exception as e:
        logger.error(f"Error creating snapshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/snapshots")
async def list_snapshots(
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Listar snapshots."""
    try:
        manager = get_snapshot_manager()
        snapshots = manager.list_snapshots(limit=limit)
        return {
            "snapshots": [
                {
                    "snapshot_id": s.snapshot_id,
                    "name": s.name,
                    "description": s.description,
                    "timestamp": s.timestamp,
                    "created_by": s.created_by
                }
                for s in snapshots
            ],
            "count": len(snapshots)
        }
    except Exception as e:
        logger.error(f"Error listing snapshots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: str) -> Dict[str, Any]:
    """Obtener snapshot."""
    try:
        manager = get_snapshot_manager()
        snapshot = manager.get_snapshot(snapshot_id)
        
        if not snapshot:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        
        return {
            "snapshot_id": snapshot.snapshot_id,
            "name": snapshot.name,
            "description": snapshot.description,
            "data": snapshot.data,
            "timestamp": snapshot.timestamp,
            "created_by": snapshot.created_by
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting snapshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/snapshots/{snapshot_id}/restore")
async def restore_snapshot(snapshot_id: str) -> Dict[str, Any]:
    """Restaurar snapshot."""
    try:
        manager = get_snapshot_manager()
        if manager.restore_snapshot(snapshot_id):
            return {"message": f"Snapshot {snapshot_id} restored successfully"}
        raise HTTPException(status_code=404, detail="Snapshot not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring snapshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/snapshots/{snapshot_id}")
async def delete_snapshot(snapshot_id: str) -> Dict[str, Any]:
    """Eliminar snapshot."""
    try:
        manager = get_snapshot_manager()
        if manager.delete_snapshot(snapshot_id):
            return {"message": f"Snapshot {snapshot_id} deleted successfully"}
        raise HTTPException(status_code=404, detail="Snapshot not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting snapshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






