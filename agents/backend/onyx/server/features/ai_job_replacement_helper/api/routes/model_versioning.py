"""
Model Versioning endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_versioning import ModelVersioningService

router = APIRouter()
versioning_service = ModelVersioningService()


@router.post("/create")
async def create_version(
    model_path: str,
    version: Optional[str] = None,
    description: str = "",
    metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Crear nueva versión de modelo"""
    try:
        model_version = versioning_service.create_version(
            model_path, version, description, metrics
        )
        
        return {
            "version": model_version.version,
            "model_path": model_version.model_path,
            "created_at": model_version.created_at.isoformat(),
            "checksum": model_version.checksum,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_versions(
    tag: Optional[str] = None,
    sort_by: str = "created_at"
) -> Dict[str, Any]:
    """Listar versiones"""
    try:
        versions = versioning_service.list_versions(tag, sort_by)
        
        return {
            "versions": [
                {
                    "version": v.version,
                    "created_at": v.created_at.isoformat(),
                    "description": v.description,
                    "metrics": v.metrics,
                }
                for v in versions
            ],
            "total": len(versions),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare")
async def compare_versions(
    version1: str,
    version2: str
) -> Dict[str, Any]:
    """Comparar dos versiones"""
    try:
        comparison = versioning_service.compare_versions(version1, version2)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




