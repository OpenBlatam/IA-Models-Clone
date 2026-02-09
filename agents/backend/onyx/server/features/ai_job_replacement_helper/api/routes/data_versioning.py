"""
Data Versioning endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_versioning import DataVersioningService, ChangeType

router = APIRouter()
versioning_service = DataVersioningService()


@router.post("/create-version")
async def create_version(
    resource_type: str,
    resource_id: str,
    data: Dict[str, Any],
    change_type: str,
    changed_by: Optional[str] = None
) -> Dict[str, Any]:
    """Crear nueva versión"""
    try:
        change_type_enum = ChangeType(change_type)
        version = versioning_service.create_version(
            resource_type, resource_id, data, change_type_enum, changed_by
        )
        return {
            "version_id": version.version_id,
            "resource_type": version.resource_type,
            "resource_id": version.resource_id,
            "change_type": version.change_type.value,
            "created_at": version.created_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{resource_type}/{resource_id}")
async def get_version_history(
    resource_type: str,
    resource_id: str
) -> Dict[str, Any]:
    """Obtener historial de versiones"""
    try:
        history = versioning_service.get_version_history(resource_type, resource_id)
        return {
            "resource_type": history.resource_type,
            "resource_id": history.resource_id,
            "total_versions": history.total_versions,
            "versions": [
                {
                    "version_id": v.version_id,
                    "change_type": v.change_type.value,
                    "created_at": v.created_at.isoformat(),
                    "changed_by": v.changed_by,
                }
                for v in history.versions
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore/{resource_type}/{resource_id}")
async def restore_version(
    resource_type: str,
    resource_id: str,
    version_id: str,
    restored_by: str
) -> Dict[str, Any]:
    """Restaurar a una versión específica"""
    try:
        version = versioning_service.restore_version(
            resource_type, resource_id, version_id, restored_by
        )
        return {
            "version_id": version.version_id,
            "restored": True,
            "restored_by": restored_by,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




