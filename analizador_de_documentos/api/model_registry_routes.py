"""
Rutas para Model Registry
===========================

Endpoints para registro de modelos.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_registry import (
    get_model_registry,
    ModelRegistry,
    ModelStage
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/model-registry",
    tags=["Model Registry"]
)


class RegisterModelRequest(BaseModel):
    """Request para registrar modelo"""
    name: str = Field(..., description="Nombre")
    description: str = Field("", description="Descripción")


class RegisterVersionRequest(BaseModel):
    """Request para registrar versión"""
    version: str = Field(..., description="Versión")
    metrics: Dict[str, float] = Field(..., description="Métricas")
    stage: str = Field("none", description="Etapa")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")


@router.post("/models")
async def register_model(
    request: RegisterModelRequest,
    system: ModelRegistry = Depends(get_model_registry)
):
    """Registrar modelo"""
    try:
        model = system.register_model(request.name, request.description)
        
        return {
            "model_id": model.model_id,
            "name": model.name,
            "description": model.description,
            "latest_version": model.latest_version
        }
    except Exception as e:
        logger.error(f"Error registrando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/versions")
async def register_version(
    model_id: str,
    request: RegisterVersionRequest,
    system: ModelRegistry = Depends(get_model_registry)
):
    """Registrar versión de modelo"""
    try:
        stage = ModelStage(request.stage)
        version = system.register_version(
            model_id, request.version, request.metrics, stage, request.metadata
        )
        
        return {
            "version_id": version.version_id,
            "model_id": version.model_id,
            "version": version.version,
            "stage": version.stage.value,
            "metrics": version.metrics
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error registrando versión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/versions/{version}/transition")
async def transition_stage(
    model_id: str,
    version: str,
    new_stage: str = Field(..., description="Nueva etapa"),
    system: ModelRegistry = Depends(get_model_registry)
):
    """Transicionar etapa de modelo"""
    try:
        stage = ModelStage(new_stage)
        model_version = system.transition_stage(model_id, version, stage)
        
        return {
            "version_id": model_version.version_id,
            "model_id": model_version.model_id,
            "version": model_version.version,
            "stage": model_version.stage.value
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error transicionando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


