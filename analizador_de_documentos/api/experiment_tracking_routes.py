"""
Rutas para Experiment Tracking
================================

Endpoints para seguimiento de experimentos.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.experiment_tracking import (
    get_experiment_tracking,
    ExperimentTracking
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/experiment-tracking",
    tags=["Experiment Tracking"]
)


class CreateExperimentRequest(BaseModel):
    """Request para crear experimento"""
    name: str = Field(..., description="Nombre")
    description: str = Field("", description="Descripción")
    tags: Optional[List[str]] = Field(None, description="Tags")


class LogRunRequest(BaseModel):
    """Request para registrar ejecución"""
    parameters: Dict[str, Any] = Field(..., description="Parámetros")
    metrics: Dict[str, float] = Field(..., description="Métricas")
    artifacts: Optional[List[str]] = Field(None, description="Artifacts")


@router.post("/experiments")
async def create_experiment(
    request: CreateExperimentRequest,
    system: ExperimentTracking = Depends(get_experiment_tracking)
):
    """Crear experimento"""
    try:
        experiment = system.create_experiment(
            request.name,
            request.description,
            request.tags
        )
        
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "tags": experiment.tags,
            "status": experiment.status.value
        }
    except Exception as e:
        logger.error(f"Error creando experimento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/runs")
async def log_run(
    experiment_id: str,
    request: LogRunRequest,
    system: ExperimentTracking = Depends(get_experiment_tracking)
):
    """Registrar ejecución"""
    try:
        run = system.log_run(
            experiment_id,
            request.parameters,
            request.metrics,
            request.artifacts
        )
        
        return {
            "run_id": run.run_id,
            "experiment_id": run.experiment_id,
            "parameters": run.parameters,
            "metrics": run.metrics,
            "status": run.status.value
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error registrando ejecución: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_experiments(
    experiment_ids: List[str] = Field(..., description="IDs de experimentos"),
    metric: str = Field("accuracy", description="Métrica"),
    system: ExperimentTracking = Depends(get_experiment_tracking)
):
    """Comparar experimentos"""
    try:
        comparison = system.compare_experiments(experiment_ids, metric)
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


