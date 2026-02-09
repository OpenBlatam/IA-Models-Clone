"""
Rutas para AutoML
==================

Endpoints para machine learning automatizado.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.automl import get_automl_system, AutoMLSystem, AutoMLTask

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/automl",
    tags=["AutoML"]
)


class CreateExperimentRequest(BaseModel):
    """Request para crear experimento"""
    task: str = Field(..., description="Tipo de tarea")
    dataset_info: Dict[str, Any] = Field(..., description="Información del dataset")
    experiment_id: Optional[str] = Field(None, description="ID del experimento")


@router.post("/experiments")
async def create_experiment(
    request: CreateExperimentRequest,
    system: AutoMLSystem = Depends(get_automl_system)
):
    """Crear experimento de AutoML"""
    try:
        task = AutoMLTask(request.task)
        experiment = system.create_experiment(
            task,
            request.dataset_info,
            request.experiment_id
        )
        
        return {
            "status": "created",
            "experiment_id": experiment.experiment_id,
            "task": experiment.task.value,
            "status": experiment.status
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Tarea inválida: {request.task}")
    except Exception as e:
        logger.error(f"Error creando experimento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/run")
async def run_experiment(
    experiment_id: str,
    system: AutoMLSystem = Depends(get_automl_system)
):
    """Ejecutar experimento de AutoML"""
    try:
        results = system.run_experiment(experiment_id)
        
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error ejecutando experimento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments(
    system: AutoMLSystem = Depends(get_automl_system)
):
    """Listar todos los experimentos"""
    experiments = system.list_experiments()
    return {"experiments": experiments}


@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    system: AutoMLSystem = Depends(get_automl_system)
):
    """Obtener experimento específico"""
    experiment = system.get_experiment(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experimento no encontrado")
    
    return {
        "experiment_id": experiment.experiment_id,
        "task": experiment.task.value,
        "status": experiment.status,
        "best_model": experiment.best_model,
        "best_score": experiment.best_score,
        "created_at": experiment.created_at
    }














