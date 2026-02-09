"""
Hyperparameter Tuning System - Sistema de optimización de hiperparámetros
===========================================================================
Optimización bayesiana, grid search, y random search
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import itertools

try:
    from optuna import create_study, Trial, Study
    from optuna.samplers import TPESampler, RandomSampler, GridSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available, using basic search")

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """Estrategias de búsqueda"""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


@dataclass
class HyperparameterSpace:
    """Espacio de búsqueda de hiperparámetros"""
    learning_rate: tuple = (1e-5, 1e-3)  # (min, max) o lista de valores
    batch_size: tuple = (16, 64)  # (min, max) o lista de valores
    num_epochs: tuple = (5, 20)  # (min, max) o lista de valores
    weight_decay: tuple = (1e-6, 1e-4)  # (min, max) o lista de valores
    dropout: tuple = (0.1, 0.5)  # (min, max) o lista de valores


class HyperparameterTuner:
    """Sistema de optimización de hiperparámetros"""
    
    def __init__(self, strategy: SearchStrategy = SearchStrategy.BAYESIAN):
        self.strategy = strategy
        self.study: Optional[Study] = None
        self.trials: List[Dict[str, Any]] = []
    
    def create_study(
        self,
        direction: str = "minimize",
        study_name: Optional[str] = None
    ):
        """Crea estudio de optimización"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using basic search")
            return
        
        sampler = None
        if self.strategy == SearchStrategy.BAYESIAN:
            sampler = TPESampler()
        elif self.strategy == SearchStrategy.RANDOM:
            sampler = RandomSampler()
        
        self.study = create_study(
            direction=direction,
            sampler=sampler,
            study_name=study_name
        )
        logger.info(f"Created study with strategy: {self.strategy}")
    
    def suggest_hyperparameters(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        """Sugiere hiperparámetros"""
        if not OPTUNA_AVAILABLE or not self.study:
            # Fallback a valores por defecto
            return {
                "learning_rate": 1e-4,
                "batch_size": 32,
                "num_epochs": 10,
                "weight_decay": 1e-5,
                "dropout": 0.2
            }
        
        if trial is None:
            trial = self.study.ask()
        
        hyperparams = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_int("batch_size", 16, 64, step=8),
            "num_epochs": trial.suggest_int("num_epochs", 5, 20),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5)
        }
        
        return hyperparams
    
    def grid_search(
        self,
        space: HyperparameterSpace,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: Optional[int] = None
    ) -> Dict[str, Any]:
        """Grid search de hiperparámetros"""
        # Convertir espacios a listas
        param_grids = {}
        
        for param_name, param_range in space.__dict__.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Rango continuo - generar valores
                if param_name in ["learning_rate", "weight_decay"]:
                    # Log scale
                    param_grids[param_name] = np.logspace(
                        np.log10(param_range[0]),
                        np.log10(param_range[1]),
                        num=5
                    ).tolist()
                else:
                    # Linear scale
                    param_grids[param_name] = np.linspace(
                        param_range[0],
                        param_range[1],
                        num=5
                    ).tolist()
            elif isinstance(param_range, list):
                param_grids[param_name] = param_range
        
        # Generar todas las combinaciones
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        best_score = float('inf')
        best_params = None
        
        total_combinations = np.prod([len(v) for v in param_values])
        if n_trials:
            total_combinations = min(total_combinations, n_trials)
        
        logger.info(f"Grid search: {total_combinations} combinations")
        
        for i, combination in enumerate(itertools.product(*param_values)):
            if n_trials and i >= n_trials:
                break
            
            params = dict(zip(param_names, combination))
            score = objective_fn(params)
            
            self.trials.append({
                "params": params,
                "score": score,
                "trial": i
            })
            
            if score < best_score:
                best_score = score
                best_params = params
            
            logger.info(f"Trial {i+1}/{total_combinations}: score={score:.4f}")
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": len(self.trials)
        }
    
    def optimize(
        self,
        objective_fn: Callable[[Trial], float],
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Optimiza hiperparámetros usando Optuna"""
        if not OPTUNA_AVAILABLE or not self.study:
            raise ValueError("Optuna not available or study not created")
        
        self.study.optimize(objective_fn, n_trials=n_trials)
        
        return {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials)
        }
    
    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """Obtiene el mejor trial"""
        if not self.trials:
            return None
        
        best_trial = min(self.trials, key=lambda x: x["score"])
        return best_trial




