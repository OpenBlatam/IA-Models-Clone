"""
Optimización automática de hiperparámetros
"""

import logging
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from dataclasses import dataclass
import optuna

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Espacio de búsqueda de hiperparámetros"""
    learning_rate: tuple = (1e-6, 1e-3)  # (min, max) en log scale
    batch_size: List[int] = None  # Lista de valores
    num_epochs: int = 10
    weight_decay: tuple = (1e-6, 1e-2)
    dropout: tuple = (0.0, 0.5)


class HyperparameterOptimizer:
    """Optimizador de hiperparámetros"""
    
    def __init__(
        self,
        direction: str = "minimize",  # "minimize" or "maximize"
        n_trials: int = 50,
        study_name: Optional[str] = None
    ):
        self.direction = direction
        self.n_trials = n_trials
        self.study_name = study_name or "hyperparameter_optimization"
        self.study = None
    
    def create_study(self):
        """Crea estudio de Optuna"""
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler()
        )
    
    def optimize(
        self,
        objective_fn: Callable,
        hyperparameter_space: Optional[HyperparameterSpace] = None
    ) -> Dict[str, Any]:
        """
        Optimiza hiperparámetros
        
        Args:
            objective_fn: Función objetivo que recibe trial y retorna métrica
            hyperparameter_space: Espacio de búsqueda
            
        Returns:
            Mejores hiperparámetros
        """
        if not self.study:
            self.create_study()
        
        def objective(trial):
            # Sugerir hiperparámetros
            if hyperparameter_space:
                lr = trial.suggest_float(
                    "learning_rate",
                    hyperparameter_space.learning_rate[0],
                    hyperparameter_space.learning_rate[1],
                    log=True
                )
                batch_size = trial.suggest_categorical(
                    "batch_size",
                    hyperparameter_space.batch_size or [8, 16, 32, 64]
                )
                weight_decay = trial.suggest_float(
                    "weight_decay",
                    hyperparameter_space.weight_decay[0],
                    hyperparameter_space.weight_decay[1],
                    log=True
                )
                dropout = trial.suggest_float(
                    "dropout",
                    hyperparameter_space.dropout[0],
                    hyperparameter_space.dropout[1]
                )
            else:
                lr = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
                batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
                weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
                dropout = trial.suggest_float("dropout", 0.0, 0.5)
            
            # Ejecutar función objetivo
            metric = objective_fn(trial, {
                "learning_rate": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "dropout": dropout
            })
            
            return metric
        
        # Optimizar
        self.study.optimize(objective, n_trials=self.n_trials)
        
        return {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": self.n_trials
        }
    
    def get_trials_dataframe(self):
        """Obtiene DataFrame con todos los trials"""
        if not self.study:
            return None
        return self.study.trials_dataframe()




