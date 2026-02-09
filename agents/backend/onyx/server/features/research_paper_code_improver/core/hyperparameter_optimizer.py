"""
Hyperparameter Optimizer - Optimizador de hiperparámetros
=========================================================
"""

import logging
import torch
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Estrategias de optimización"""
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    TPE = "tpe"  # Tree-structured Parzen Estimator


@dataclass
class HyperparameterSpace:
    """Espacio de hiperparámetros"""
    name: str
    param_type: str  # "float", "int", "categorical"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False


@dataclass
class TrialResult:
    """Resultado de un trial"""
    trial_id: int
    hyperparameters: Dict[str, Any]
    metric_value: float
    status: str = "completed"  # completed, failed, pruned
    duration: float = 0.0


class HyperparameterOptimizer:
    """Optimizador de hiperparámetros"""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH):
        self.strategy = strategy
        self.search_space: Dict[str, HyperparameterSpace] = {}
        self.trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None
    
    def add_hyperparameter(
        self,
        name: str,
        param_type: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        choices: Optional[List[Any]] = None,
        log_scale: bool = False
    ):
        """Agrega un hiperparámetro al espacio de búsqueda"""
        space = HyperparameterSpace(
            name=name,
            param_type=param_type,
            min_value=min_value,
            max_value=max_value,
            choices=choices,
            log_scale=log_scale
        )
        self.search_space[name] = space
    
    def sample_hyperparameters(self) -> Dict[str, Any]:
        """Muestra hiperparámetros según la estrategia"""
        if self.strategy == OptimizationStrategy.RANDOM_SEARCH:
            return self._random_search_sample()
        elif self.strategy == OptimizationStrategy.GRID_SEARCH:
            return self._grid_search_sample()
        else:
            return self._random_search_sample()
    
    def _random_search_sample(self) -> Dict[str, Any]:
        """Muestra aleatoria"""
        params = {}
        for name, space in self.search_space.items():
            if space.param_type == "float":
                if space.log_scale:
                    min_val = np.log10(space.min_value) if space.min_value else -3
                    max_val = np.log10(space.max_value) if space.max_value else 3
                    value = 10 ** random.uniform(min_val, max_val)
                else:
                    value = random.uniform(space.min_value or 0, space.max_value or 1)
                params[name] = value
            elif space.param_type == "int":
                params[name] = random.randint(
                    int(space.min_value or 0),
                    int(space.max_value or 100)
                )
            elif space.param_type == "categorical":
                params[name] = random.choice(space.choices or [])
        return params
    
    def _grid_search_sample(self) -> Dict[str, Any]:
        """Muestra de grid search (simplificado)"""
        # En producción, implementar grid completo
        return self._random_search_sample()
    
    def optimize(
        self,
        objective_function: Callable,
        n_trials: int = 100,
        direction: str = "minimize"  # "minimize" or "maximize"
    ) -> TrialResult:
        """Optimiza hiperparámetros"""
        best_value = float('inf') if direction == "minimize" else float('-inf')
        
        for trial_id in range(n_trials):
            # Sample hyperparameters
            hyperparameters = self.sample_hyperparameters()
            
            try:
                # Evaluate objective
                metric_value = objective_function(hyperparameters)
                
                # Create trial result
                trial = TrialResult(
                    trial_id=trial_id,
                    hyperparameters=hyperparameters,
                    metric_value=metric_value
                )
                self.trials.append(trial)
                
                # Update best
                is_better = (
                    (direction == "minimize" and metric_value < best_value) or
                    (direction == "maximize" and metric_value > best_value)
                )
                
                if is_better:
                    best_value = metric_value
                    self.best_trial = trial
                    logger.info(f"Trial {trial_id}: Nuevo mejor valor = {best_value}")
            
            except Exception as e:
                logger.error(f"Error en trial {trial_id}: {e}")
                trial = TrialResult(
                    trial_id=trial_id,
                    hyperparameters=hyperparameters,
                    metric_value=float('inf') if direction == "minimize" else float('-inf'),
                    status="failed"
                )
                self.trials.append(trial)
        
        return self.best_trial
    
    def get_best_hyperparameters(self) -> Optional[Dict[str, Any]]:
        """Obtiene los mejores hiperparámetros"""
        if self.best_trial:
            return self.best_trial.hyperparameters
        return None
    
    def get_trials_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de trials"""
        if not self.trials:
            return {}
        
        completed_trials = [t for t in self.trials if t.status == "completed"]
        if not completed_trials:
            return {}
        
        metric_values = [t.metric_value for t in completed_trials]
        
        return {
            "total_trials": len(self.trials),
            "completed_trials": len(completed_trials),
            "failed_trials": len([t for t in self.trials if t.status == "failed"]),
            "best_metric": min(metric_values) if metric_values else None,
            "worst_metric": max(metric_values) if metric_values else None,
            "mean_metric": np.mean(metric_values) if metric_values else None,
            "std_metric": np.std(metric_values) if metric_values else None
        }




