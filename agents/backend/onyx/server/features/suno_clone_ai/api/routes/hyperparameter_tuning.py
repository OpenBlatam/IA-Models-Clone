"""
API de Hyperparameter Tuning

Endpoints para:
- Grid search
- Random search
- Obtener mejores hiperparámetros
- Historial de trials
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.hyperparameter_tuner import (
    get_hyperparameter_tuner,
    HyperparameterConfig,
    SearchStrategy
)
from middleware.auth_middleware import require_role

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/hyperparameter-tuning",
    tags=["hyperparameter-tuning"],
    dependencies=[Depends(require_role("admin"))]  # Requiere rol admin
)


@router.post("/grid-search")
async def run_grid_search(
    hyperparameters: Dict[str, Dict[str, Any]] = Body(..., description="Configuración de hiperparámetros"),
    max_trials: Optional[int] = Body(None, description="Máximo de trials")
) -> Dict[str, Any]:
    """
    Ejecuta grid search (requiere rol admin).
    
    Nota: En producción, esto ejecutaría trials reales.
    """
    try:
        # Convertir configuración
        configs = {}
        for name, config_data in hyperparameters.items():
            configs[name] = HyperparameterConfig(
                name=name,
                type=config_data.get("type", "choice"),
                values=config_data.get("values", []),
                min_value=config_data.get("min_value"),
                max_value=config_data.get("max_value"),
                step=config_data.get("step")
            )
        
        tuner = get_hyperparameter_tuner()
        
        # Función objetivo de ejemplo (en producción sería real)
        def objective_func(params: Dict[str, Any]) -> Dict[str, float]:
            # Simular evaluación
            import random
            return {
                "score": random.uniform(0.5, 1.0),
                "loss": random.uniform(0.0, 0.5)
            }
        
        best_trial = tuner.grid_search(
            hyperparameters=configs,
            objective_func=objective_func,
            max_trials=max_trials
        )
        
        return {
            "message": "Grid search completed",
            "best_trial": {
                "hyperparameters": best_trial.hyperparameters if best_trial else None,
                "metrics": best_trial.metrics if best_trial else None
            },
            "total_trials": len(tuner.trials)
        }
    except Exception as e:
        logger.error(f"Error running grid search: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running grid search: {str(e)}"
        )


@router.post("/random-search")
async def run_random_search(
    hyperparameters: Dict[str, Dict[str, Any]] = Body(..., description="Configuración de hiperparámetros"),
    n_trials: int = Body(50, ge=1, le=1000, description="Número de trials")
) -> Dict[str, Any]:
    """
    Ejecuta random search (requiere rol admin).
    """
    try:
        # Similar a grid search
        configs = {}
        for name, config_data in hyperparameters.items():
            configs[name] = HyperparameterConfig(
                name=name,
                type=config_data.get("type", "choice"),
                values=config_data.get("values", []),
                min_value=config_data.get("min_value"),
                max_value=config_data.get("max_value"),
                step=config_data.get("step")
            )
        
        tuner = get_hyperparameter_tuner()
        
        def objective_func(params: Dict[str, Any]) -> Dict[str, float]:
            import random
            return {
                "score": random.uniform(0.5, 1.0),
                "loss": random.uniform(0.0, 0.5)
            }
        
        best_trial = tuner.random_search(
            hyperparameters=configs,
            objective_func=objective_func,
            n_trials=n_trials
        )
        
        return {
            "message": "Random search completed",
            "best_trial": {
                "hyperparameters": best_trial.hyperparameters if best_trial else None,
                "metrics": best_trial.metrics if best_trial else None
            },
            "total_trials": len(tuner.trials)
        }
    except Exception as e:
        logger.error(f"Error running random search: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running random search: {str(e)}"
        )


@router.get("/best")
async def get_best_hyperparameters() -> Dict[str, Any]:
    """
    Obtiene los mejores hiperparámetros encontrados.
    """
    try:
        tuner = get_hyperparameter_tuner()
        best_params = tuner.get_best_hyperparameters()
        
        if not best_params:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No trials completed yet"
            )
        
        return {
            "best_hyperparameters": best_params,
            "best_trial": {
                "hyperparameters": tuner.best_trial.hyperparameters if tuner.best_trial else None,
                "metrics": tuner.best_trial.metrics if tuner.best_trial else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting best hyperparameters: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving best hyperparameters: {str(e)}"
        )


@router.get("/history")
async def get_trial_history() -> Dict[str, Any]:
    """
    Obtiene el historial completo de trials.
    """
    try:
        tuner = get_hyperparameter_tuner()
        history = tuner.get_trial_history()
        
        return {
            "trials": history,
            "total": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting trial history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving history: {str(e)}"
        )


@router.get("/summary")
async def get_tuning_summary() -> Dict[str, Any]:
    """
    Obtiene un resumen del proceso de tuning.
    """
    try:
        tuner = get_hyperparameter_tuner()
        summary = tuner.get_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting tuning summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving summary: {str(e)}"
        )

