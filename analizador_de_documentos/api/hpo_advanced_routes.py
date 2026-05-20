"""
Rutas para Advanced Hyperparameter Optimization
================================================

Endpoints para optimización avanzada de hiperparámetros.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_hyperparameter_optimization import (
    get_advanced_hyperparameter_optimization,
    AdvancedHyperparameterOptimization,
    OptimizationMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/advanced-hyperparameter-optimization",
    tags=["Advanced Hyperparameter Optimization"]
)


class CreateExperimentRequest(BaseModel):
    """Request para crear experimento"""
    search_space: Dict[str, Any] = Field(..., description="Espacio de búsqueda")
    method: str = Field("bayesian", description="Método")


@router.post("/experiments")
async def create_experiment(
    request: CreateExperimentRequest,
    system: AdvancedHyperparameterOptimization = Depends(get_advanced_hyperparameter_optimization)
):
    """Crear experimento de optimización"""
    try:
        method = OptimizationMethod(request.method)
        experiment = system.create_experiment(request.search_space, method)
        
        return {
            "experiment_id": experiment.experiment_id,
            "method": experiment.method.value,
            "search_space": experiment.search_space,
            "status": experiment.status.value
        }
    except Exception as e:
        logger.error(f"Error creando experimento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/optimize")
async def optimize(
    experiment_id: str,
    max_trials: int = Field(100, description="Máximo de trials"),
    objective: str = Field("accuracy", description="Objetivo"),
    system: AdvancedHyperparameterOptimization = Depends(get_advanced_hyperparameter_optimization)
):
    """Optimizar hiperparámetros"""
    try:
        best_config = system.optimize(experiment_id, max_trials, objective)
        
        return {
            "config_id": best_config.config_id,
            "hyperparameters": best_config.hyperparameters,
            "performance": best_config.performance,
            "training_time": best_config.training_time
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/importance")
async def analyze_importance(
    experiment_id: str,
    system: AdvancedHyperparameterOptimization = Depends(get_advanced_hyperparameter_optimization)
):
    """Analizar importancia de hiperparámetros"""
    try:
        importance = system.analyze_hyperparameter_importance(experiment_id)
        
        return {"experiment_id": experiment_id, "importance": importance}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analizando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


