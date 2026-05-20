"""
Rutas para Versionado de Modelos
==================================

Endpoints para gestión de versiones de modelos.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_versioning import (
    get_model_version_manager,
    ModelVersionManager,
    ModelStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/versions",
    tags=["Model Versioning"]
)


class RegisterVersionRequest(BaseModel):
    """Request para registrar versión"""
    model_name: str = Field(..., description="Nombre del modelo")
    model_path: str = Field(..., description="Ruta al modelo")
    version: Optional[str] = Field(None, description="Versión (auto si None)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos")
    tags: Optional[list[str]] = Field(None, description="Tags")


@router.post("/")
async def register_version(
    request: RegisterVersionRequest,
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Registrar nueva versión de modelo"""
    try:
        version = manager.register_version(
            request.model_name,
            request.model_path,
            request.version,
            request.metadata,
            request.tags
        )
        
        return {
            "status": "registered",
            "version": {
                "version": version.version,
                "model_name": version.model_name,
                "status": version.status.value,
                "created_at": version.created_at
            }
        }
    except Exception as e:
        logger.error(f"Error registrando versión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_versions(
    model_name: Optional[str] = None,
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Listar versiones"""
    versions = manager.list_versions(model_name)
    return {"versions": versions}


@router.get("/{model_name}/{version}")
async def get_version(
    model_name: str,
    version: str,
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Obtener versión específica"""
    version_obj = manager.get_version(model_name, version)
    if not version_obj:
        raise HTTPException(status_code=404, detail="Versión no encontrada")
    
    return {
        "version": version_obj.version,
        "model_name": version_obj.model_name,
        "status": version_obj.status.value,
        "created_at": version_obj.created_at,
        "metadata": version_obj.metadata,
        "performance_metrics": version_obj.performance_metrics,
        "tags": version_obj.tags
    }


@router.post("/{model_name}/{version}/deploy")
async def deploy_version(
    model_name: str,
    version: str,
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Desplegar versión"""
    success = manager.deploy_version(model_name, version)
    if not success:
        raise HTTPException(status_code=404, detail="Versión no encontrada")
    
    return {"status": "deployed", "model_name": model_name, "version": version}


@router.get("/{model_name}/deployed")
async def get_deployed_version(
    model_name: str,
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Obtener versión desplegada"""
    version = manager.get_deployed_version(model_name)
    if not version:
        raise HTTPException(status_code=404, detail="No hay versión desplegada")
    
    return {
        "version": version.version,
        "status": version.status.value,
        "created_at": version.created_at
    }


@router.get("/{model_name}/compare")
async def compare_versions(
    model_name: str,
    version1: str,
    version2: str,
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Comparar dos versiones"""
    comparison = manager.compare_versions(model_name, version1, version2)
    return comparison
