"""
Rutas para ML Pipeline Orchestration
======================================

Endpoints para orquestación de pipelines.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.ml_pipeline_orchestration import (
    get_ml_pipeline_orchestration,
    MLPipelineOrchestration
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/ml-pipeline-orchestration",
    tags=["ML Pipeline Orchestration"]
)


class CreatePipelineRequest(BaseModel):
    """Request para crear pipeline"""
    name: str = Field(..., description="Nombre")
    description: str = Field("", description="Descripción")
    steps: Optional[List[Dict[str, Any]]] = Field(None, description="Pasos")


@router.post("/pipelines")
async def create_pipeline(
    request: CreatePipelineRequest,
    system: MLPipelineOrchestration = Depends(get_ml_pipeline_orchestration)
):
    """Crear pipeline"""
    try:
        pipeline = system.create_pipeline(
            request.name,
            request.description,
            request.steps
        )
        
        return {
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "description": pipeline.description,
            "status": pipeline.status,
            "steps": [
                {
                    "step_id": s.step_id,
                    "name": s.name,
                    "stage": s.stage.value
                }
                for s in pipeline.steps
            ]
        }
    except Exception as e:
        logger.error(f"Error creando pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipelines/{pipeline_id}/execute")
async def execute_pipeline(
    pipeline_id: str,
    system: MLPipelineOrchestration = Depends(get_ml_pipeline_orchestration)
):
    """Ejecutar pipeline"""
    try:
        pipeline = system.execute_pipeline(pipeline_id)
        
        return {
            "pipeline_id": pipeline.pipeline_id,
            "status": pipeline.status,
            "started_at": pipeline.started_at,
            "completed_at": pipeline.completed_at
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error ejecutando pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines/{pipeline_id}/status")
async def get_pipeline_status(
    pipeline_id: str,
    system: MLPipelineOrchestration = Depends(get_ml_pipeline_orchestration)
):
    """Obtener estado del pipeline"""
    try:
        status = system.get_pipeline_status(pipeline_id)
        
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


