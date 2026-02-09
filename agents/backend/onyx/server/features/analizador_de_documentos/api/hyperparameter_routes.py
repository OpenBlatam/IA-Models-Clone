"""
Rutas para Hyperparameter Optimization
========================================

Endpoints para optimización de hiperparámetros.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.hyperparameter_optimization import (
    get_hyperparameter_optimizer,
    HyperparameterOptimizer,
    OptimizationMethod,
    HyperparameterConfig
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/hyperparameter-optimization",
    tags=["Hyperparameter Optimization"]
)


class CreateOptimizationRequest(BaseModel):
    """Request para crear optimización"""
    hyperparameters: List[Dict[str, Any]] = Field(..., description="Hiperparámetros")
    method: str = Field("bayesian", description="Método")


class OptimizeRequest(BaseModel):
    """Request para optimizar"""
    max_trials: int = Field(100, description="Número máximo de trials")
    method: str = Field("bayesian", description="Método")


@router.post("/create")
async def create_optimization(
    request: CreateOptimizationRequest,
    optimizer: HyperparameterOptimizer = Depends(get_hyperparameter_optimizer)
):
    """Crear optimización de hiperparámetros"""
    try:
        method = OptimizationMethod(request.method)
        hyperparams = [
            HyperparameterConfig(
                param_name=hp.get("param_name", ""),
                param_type=hp.get("param_type", "float"),
                search_space=hp.get("search_space", {})
            )
            for hp in request.hyperparameters
        ]
        
        optimization_id = optimizer.create_optimization(hyperparams, method)
        
        return {"optimization_id": optimization_id, "method": method.value}
    except Exception as e:
        logger.error(f"Error creando optimización: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/{optimization_id}")
async def optimize(
    optimization_id: str,
    request: OptimizeRequest,
    optimizer: HyperparameterOptimizer = Depends(get_hyperparameter_optimizer)
):
    """Ejecutar optimización"""
    try:
        method = OptimizationMethod(request.method)
        # En producción, se pasaría la función objetivo real
        result = optimizer.optimize(optimization_id, None, request.max_trials, method)
        
        return {
            "optimization_id": result.optimization_id,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "method": result.method.value,
            "trials": result.trials
        }
    except Exception as e:
        logger.error(f"Error optimizando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best-params/{optimization_id}")
async def get_best_params(
    optimization_id: str,
    optimizer: HyperparameterOptimizer = Depends(get_hyperparameter_optimizer)
):
    """Obtener mejores hiperparámetros"""
    try:
        best_params = optimizer.get_best_params(optimization_id)
        
        return {"optimization_id": optimization_id, "best_params": best_params}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo parámetros: {e}")
        raise HTTPException(status_code=500, detail=str(e))


