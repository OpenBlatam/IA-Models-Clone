"""
Rutas para Pipeline de ML
===========================

Endpoints para gestión de pipelines de machine learning.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.ml_pipeline import MLPipeline, PipelineStage

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/pipelines",
    tags=["ML Pipeline"]
)


class PipelineStepRequest(BaseModel):
    """Request para agregar paso"""
    name: str
    stage: str
    dependencies: List[str] = Field(default_factory=list)


@router.post("/")
async def create_pipeline(
    pipeline_id: str = None
):
    """Crear nuevo pipeline"""
    try:
        pipeline = MLPipeline(pipeline_id)
        return {
            "status": "created",
            "pipeline_id": pipeline.pipeline_id,
            "info": pipeline.get_pipeline_info()
        }
    except Exception as e:
        logger.error(f"Error creando pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pipeline_id}/steps")
async def add_step(
    pipeline_id: str,
    step_request: PipelineStepRequest,
    pipelines: Dict[str, MLPipeline] = Depends(lambda: {})
):
    """Agregar paso al pipeline"""
    # En producción, los pipelines se almacenarían en una base de datos
    # Por ahora, usamos un diccionario temporal
    if pipeline_id not in pipelines:
        pipelines[pipeline_id] = MLPipeline(pipeline_id)
    
    pipeline = pipelines[pipeline_id]
    
    try:
        stage = PipelineStage(step_request.stage)
        # Nota: En producción, la función se pasaría de otra manera
        # Por ahora, esto es solo un ejemplo
        pipeline.add_step(
            step_request.name,
            stage,
            lambda ctx: None,  # Placeholder
            step_request.dependencies
        )
        
        return {
            "status": "added",
            "pipeline_id": pipeline_id,
            "step": step_request.name
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Stage inválido")
    except Exception as e:
        logger.error(f"Error agregando paso: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}")
async def get_pipeline_info(
    pipeline_id: str,
    pipelines: Dict[str, MLPipeline] = Depends(lambda: {})
):
    """Obtener información del pipeline"""
    if pipeline_id not in pipelines:
        raise HTTPException(status_code=404, detail="Pipeline no encontrado")
    
    return pipelines[pipeline_id].get_pipeline_info()
















