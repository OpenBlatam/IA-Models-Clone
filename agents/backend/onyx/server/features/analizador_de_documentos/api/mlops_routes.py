"""
Rutas para MLOps Completo
===========================

Endpoints para MLOps completo.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.mlops_complete import (
    get_mlops,
    MLOpsComplete,
    MLOpsStage
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/mlops",
    tags=["MLOps Complete"]
)


class CreatePipelineRequest(BaseModel):
    """Request para crear pipeline"""
    stages: List[str] = Field(..., description="Etapas")


@router.post("/pipelines")
async def create_pipeline(
    model_id: str = Field(..., description="ID del modelo"),
    request: CreatePipelineRequest,
    system: MLOpsComplete = Depends(get_mlops)
):
    """Crear pipeline de ML"""
    try:
        stages = [MLOpsStage(s) for s in request.stages]
        pipeline = system.create_pipeline(model_id, stages)
        
        return {
            "pipeline_id": pipeline.pipeline_id,
            "model_id": pipeline.model_id,
            "stages": [s.value for s in pipeline.stages],
            "status": pipeline.status
        }
    except Exception as e:
        logger.error(f"Error creando pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployments")
async def deploy_model(
    model_id: str = Field(..., description="ID del modelo"),
    environment: str = Field("production", description="Ambiente"),
    version: str = Field("1.0.0", description="Versión"),
    system: MLOpsComplete = Depends(get_mlops)
):
    """Desplegar modelo"""
    try:
        deployment = system.deploy_model(model_id, environment, version)
        
        return {
            "deployment_id": deployment.deployment_id,
            "model_id": deployment.model_id,
            "environment": deployment.environment,
            "version": deployment.version,
            "status": deployment.status
        }
    except Exception as e:
        logger.error(f"Error desplegando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployments/{deployment_id}/monitor")
async def monitor_model(
    deployment_id: str,
    metrics: Dict[str, Any] = Field(..., description="Métricas"),
    system: MLOpsComplete = Depends(get_mlops)
):
    """Monitorear modelo"""
    try:
        analysis = system.monitor_model(deployment_id, metrics)
        
        return analysis
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error monitoreando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployments/{deployment_id}/rollback")
async def rollback_deployment(
    deployment_id: str,
    previous_version: str = Field(..., description="Versión anterior"),
    system: MLOpsComplete = Depends(get_mlops)
):
    """Hacer rollback"""
    try:
        result = system.rollback_deployment(deployment_id, previous_version)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error haciendo rollback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
