"""
Hyperparameter Optimization Service - Optimización de hiperparámetros
======================================================================

Sistema para optimización de hiperparámetros usando técnicas avanzadas.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import numpy as np

logger = logging.getLogger(__name__)

# Try to import optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available")


class OptimizationMethod(str, Enum):
    """Métodos de optimización"""
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"  # Optuna TPE
    EVOLUTIONARY = "evolutionary"


@dataclass
class HyperparameterSpace:
    """Espacio de búsqueda de hiperparámetros"""
    learning_rate: Dict[str, Any] = field(default_factory=lambda: {
        "type": "float",
        "low": 1e-5,
        "high": 1e-2,
        "log": True
    })
    batch_size: Dict[str, Any] = field(default_factory=lambda: {
        "type": "int",
        "choices": [16, 32, 64, 128]
    })
    num_epochs: Dict[str, Any] = field(default_factory=lambda: {
        "type": "int",
        "low": 5,
        "high": 50
    })
    dropout: Dict[str, Any] = field(default_factory=lambda: {
        "type": "float",
        "low": 0.0,
        "high": 0.5
    })


@dataclass
class OptimizationResult:
    """Resultado de optimización"""
    best_params: Dict[str, Any]
    best_score: float
    trials: List[Dict[str, Any]]
    optimization_time: float


class HyperparameterOptimizationService:
    """Servicio de optimización de hiperparámetros"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.studies: Dict[str, Any] = {}
        logger.info("HyperparameterOptimizationService initialized")
    
    def optimize_with_optuna(
        self,
        study_name: str,
        objective_function: Callable,
        space: HyperparameterSpace,
        n_trials: int = 100,
        direction: str = "minimize"
    ) -> OptimizationResult:
        """Optimizar usando Optuna"""
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available")
        
        import time
        start_time = time.time()
        
        study = optuna.create_study(
            direction=direction,
            study_name=study_name
        )
        
        def optuna_objective(trial):
            # Suggest hyperparameters
            params = {}
            
            if space.learning_rate.get("type") == "float":
                if space.learning_rate.get("log", False):
                    params["learning_rate"] = trial.suggest_float(
                        "learning_rate",
                        space.learning_rate["low"],
                        space.learning_rate["high"],
                        log=True
                    )
                else:
                    params["learning_rate"] = trial.suggest_float(
                        "learning_rate",
                        space.learning_rate["low"],
                        space.learning_rate["high"]
                    )
            
            if space.batch_size.get("type") == "int":
                if "choices" in space.batch_size:
                    params["batch_size"] = trial.suggest_categorical(
                        "batch_size",
                        space.batch_size["choices"]
                    )
                else:
                    params["batch_size"] = trial.suggest_int(
                        "batch_size",
                        space.batch_size.get("low", 16),
                        space.batch_size.get("high", 128)
                    )
            
            if space.num_epochs.get("type") == "int":
                params["num_epochs"] = trial.suggest_int(
                    "num_epochs",
                    space.num_epochs["low"],
                    space.num_epochs["high"]
                )
            
            if space.dropout.get("type") == "float":
                params["dropout"] = trial.suggest_float(
                    "dropout",
                    space.dropout["low"],
                    space.dropout["high"]
                )
            
            # Call objective function
            score = objective_function(params)
            
            return score
        
        study.optimize(optuna_objective, n_trials=n_trials)
        
        optimization_time = time.time() - start_time
        
        # Extract results
        trials = [
            {
                "params": trial.params,
                "value": trial.value,
                "number": trial.number,
            }
            for trial in study.trials
        ]
        
        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            trials=trials,
            optimization_time=optimization_time
        )
        
        self.studies[study_name] = study
        
        logger.info(f"Optimization completed: {study_name}, best score: {study.best_value}")
        
        return result
    
    def random_search(
        self,
        objective_function: Callable,
        space: HyperparameterSpace,
        n_trials: int = 50
    ) -> OptimizationResult:
        """Búsqueda aleatoria"""
        import time
        start_time = time.time()
        
        best_score = float("inf")
        best_params = None
        trials = []
        
        for i in range(n_trials):
            params = {}
            
            # Sample learning rate
            if space.learning_rate.get("log", False):
                params["learning_rate"] = 10 ** np.random.uniform(
                    np.log10(space.learning_rate["low"]),
                    np.log10(space.learning_rate["high"])
                )
            else:
                params["learning_rate"] = np.random.uniform(
                    space.learning_rate["low"],
                    space.learning_rate["high"]
                )
            
            # Sample batch size
            if "choices" in space.batch_size:
                params["batch_size"] = random.choice(space.batch_size["choices"])
            else:
                params["batch_size"] = random.randint(
                    space.batch_size.get("low", 16),
                    space.batch_size.get("high", 128)
                )
            
            # Sample num_epochs
            params["num_epochs"] = random.randint(
                space.num_epochs["low"],
                space.num_epochs["high"]
            )
            
            # Sample dropout
            params["dropout"] = np.random.uniform(
                space.dropout["low"],
                space.dropout["high"]
            )
            
            # Evaluate
            score = objective_function(params)
            
            trials.append({
                "params": params,
                "value": score,
                "number": i,
            })
            
            if score < best_score:
                best_score = score
                best_params = params
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            trials=trials,
            optimization_time=optimization_time
        )
    
    def get_study_summary(self, study_name: str) -> Dict[str, Any]:
        """Obtener resumen del estudio"""
        if not OPTUNA_AVAILABLE:
            return {"error": "Optuna not available"}
        
        study = self.studies.get(study_name)
        if not study:
            return {"error": "Study not found"}
        
        return {
            "study_name": study_name,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "direction": study.direction,
        }




