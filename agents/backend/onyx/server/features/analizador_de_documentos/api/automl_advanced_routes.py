"""
Rutas para AutoML Avanzado
===========================

Endpoints para AutoML avanzado.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.automated_ml_advanced import (
    get_automl_advanced,
    AutomatedMLAdvanced,
    AutoMLTask
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/automl-advanced",
    tags=["AutoML Advanced"]
)


class CreateExperimentRequest(BaseModel):
    """Request para crear experimento"""
    task: str = Field(..., description="Tarea")
    data: List[Dict[str, Any]] = Field(..., description="Datos")
    target: str = Field(..., description="Columna objetivo")


@router.post("/experiments")
async def create_experiment(
    request: CreateExperimentRequest,
    system: AutomatedMLAdvanced = Depends(get_automl_advanced)
):
    """Crear experimento de AutoML"""
    try:
        task = AutoMLTask(request.task)
        experiment = system.create_experiment(task, request.data, request.target)
        
        return {
            "experiment_id": experiment.experiment_id,
            "task": experiment.task.value,
            "target": experiment.target,
            "status": experiment.status
        }
    except Exception as e:
        logger.error(f"Error creando experimento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/run")
async def run_experiment(
    experiment_id: str,
    max_models: int = Field(10, description="Máximo de modelos"),
    time_limit_minutes: int = Field(60, description="Límite de tiempo"),
    system: AutomatedMLAdvanced = Depends(get_automl_advanced)
):
    """Ejecutar experimento"""
    try:
        result = system.run_experiment(experiment_id, max_models, time_limit_minutes)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error ejecutando experimento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/results")
async def get_results(
    experiment_id: str,
    system: AutomatedMLAdvanced = Depends(get_automl_advanced)
):
    """Obtener resultados del experimento"""
    try:
        results = system.get_experiment_results(experiment_id)
        
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo resultados: {e}")
        raise HTTPException(status_code=500, detail=str(e))


