"""
Rutas para Data Versioning
============================

Endpoints para versionado de datos.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.data_versioning import (
    get_data_versioning,
    DataVersioning,
    VersionType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/data-versioning",
    tags=["Data Versioning"]
)


class CreateDatasetRequest(BaseModel):
    """Request para crear dataset"""
    name: str = Field(..., description="Nombre")
    description: str = Field("", description="Descripción")


@router.post("/datasets")
async def create_dataset(
    request: CreateDatasetRequest,
    system: DataVersioning = Depends(get_data_versioning)
):
    """Crear dataset"""
    try:
        dataset = system.create_dataset(request.name, request.description)
        
        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "description": dataset.description,
            "current_version": dataset.current_version
        }
    except Exception as e:
        logger.error(f"Error creando dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_id}/versions")
async def create_version(
    dataset_id: str,
    version: str = Field(..., description="Versión"),
    version_type: str = Field("snapshot", description="Tipo"),
    description: str = Field("", description="Descripción"),
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata"),
    system: DataVersioning = Depends(get_data_versioning)
):
    """Crear versión de datos"""
    try:
        v_type = VersionType(version_type)
        data_version = system.create_version(
            dataset_id, version, v_type, description, metadata
        )
        
        return {
            "version_id": data_version.version_id,
            "dataset_id": data_version.dataset_id,
            "version": data_version.version,
            "version_type": data_version.version_type.value
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creando versión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_id}/compare")
async def compare_versions(
    dataset_id: str,
    version1: str = Field(..., description="Versión 1"),
    version2: str = Field(..., description="Versión 2"),
    system: DataVersioning = Depends(get_data_versioning)
):
    """Comparar versiones"""
    try:
        comparison = system.compare_versions(dataset_id, version1, version2)
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


