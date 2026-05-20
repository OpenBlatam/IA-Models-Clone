"""
Rutas para Versionado de Modelos
==================================

Endpoints para gestión de versiones de modelos.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_versioning import get_model_version_manager, ModelVersionManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/versions",
    tags=["Model Versioning"]
)


class RegisterVersionRequest(BaseModel):
    """Request para registrar versión"""
    version: str = Field(..., description="Versión del modelo")
    model_path: str = Field(..., description="Ruta al modelo")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Métricas de rendimiento")


@router.post("/")
async def register_version(
    request: RegisterVersionRequest,
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Registrar nueva versión de modelo"""
    try:
        version_obj = manager.register_version(
            request.version,
            request.model_path,
            request.metadata,
            request.performance_metrics
        )
        
        return {
            "status": "registered",
            "version": version_obj.version,
            "is_active": version_obj.is_active
        }
    except Exception as e:
        logger.error(f"Error registrando versión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_versions(
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Listar todas las versiones"""
    return {"versions": manager.list_versions()}


@router.get("/current")
async def get_current_version(
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Obtener versión actual"""
    version = manager.get_current_version()
    if not version:
        raise HTTPException(status_code=404, detail="No hay versión activa")
    
    return {
        "version": version.version,
        "model_path": version.model_path,
        "metadata": version.metadata,
        "performance_metrics": version.performance_metrics
    }


@router.post("/switch/{version}")
async def switch_version(
    version: str,
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Cambiar a otra versión"""
    success = manager.switch_version(version)
    if not success:
        raise HTTPException(status_code=404, detail="Versión no encontrada")
    
    return {"status": "switched", "version": version}


@router.get("/compare/{version1}/{version2}")
async def compare_versions(
    version1: str,
    version2: str,
    manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Comparar dos versiones"""
    comparison = manager.compare_versions(version1, version2)
    if "error" in comparison:
        raise HTTPException(status_code=404, detail=comparison["error"])
    
    return comparison
















