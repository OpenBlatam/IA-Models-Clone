"""
Hyperparameter Optimization endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Callable
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.hyperparameter_optimization import (
    HyperparameterOptimizationService,
    HyperparameterSpace
)

router = APIRouter()
optimization_service = HyperparameterOptimizationService()


@router.post("/optimize")
async def optimize_hyperparameters(
    study_name: str,
    method: str = "bayesian",
    n_trials: int = 100
) -> Dict[str, Any]:
    """Optimizar hiperparámetros"""
    try:
        # Create default space
        space = HyperparameterSpace()
        
        # In production, you would pass the actual objective function
        # For now, we'll use a placeholder
        def dummy_objective(params):
            # This would be your actual training function
            return 0.5  # Placeholder
        
        if method == "bayesian":
            result = optimization_service.optimize_with_optuna(
                study_name, dummy_objective, space, n_trials
            )
        elif method == "random_search":
            result = optimization_service.random_search(
                dummy_objective, space, n_trials
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            "study_name": study_name,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": len(result.trials),
            "optimization_time": result.optimization_time,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/study/{study_name}")
async def get_study_summary(study_name: str) -> Dict[str, Any]:
    """Obtener resumen del estudio"""
    try:
        summary = optimization_service.get_study_summary(study_name)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




