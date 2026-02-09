"""
Hyperparameter Auto-tuning - Auto-tuning de hiperparámetros
============================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class TuningMethod(Enum):
    """Métodos de tuning"""
    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"
    TPE = "tpe"  # Tree-structured Parzen Estimator


@dataclass
class HyperparameterSpace:
    """Espacio de hiperparámetros"""
    learning_rate: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    weight_decay: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.1])
    dropout: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])


class HyperparameterAutoTuner:
    """Auto-tuner de hiperparámetros"""
    
    def __init__(self, method: TuningMethod = TuningMethod.RANDOM):
        self.method = method
        self.tuning_history: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_score: float = 0.0
    
    def tune(
        self,
        model_builder: Callable,
        train_fn: Callable,
        eval_fn: Callable,
        hyperparameter_space: HyperparameterSpace,
        num_trials: int = 20
    ) -> Dict[str, Any]:
        """Tunea hiperparámetros"""
        search_space = {
            "learning_rate": hyperparameter_space.learning_rate,
            "batch_size": hyperparameter_space.batch_size,
            "weight_decay": hyperparameter_space.weight_decay,
            "dropout": hyperparameter_space.dropout
        }
        
        for trial in range(num_trials):
            # Generar configuración
            if self.method == TuningMethod.RANDOM:
                config = self._random_config(search_space)
            elif self.method == TuningMethod.GRID:
                config = self._grid_config(search_space, trial)
            else:
                config = self._random_config(search_space)
            
            logger.info(f"Trial {trial + 1}/{num_trials}: {config}")
            
            # Entrenar y evaluar
            try:
                model = model_builder(config)
                train_fn(model, config)
                score = eval_fn(model)
                
                self.tuning_history.append({
                    "trial": trial,
                    "config": config,
                    "score": score
                })
                
                # Actualizar mejor
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = config
                    logger.info(f"Nuevo mejor score: {score:.4f}")
            
            except Exception as e:
                logger.warning(f"Trial {trial + 1} falló: {e}")
                continue
        
        return self.best_config or {}
    
    def _random_config(self, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Configuración aleatoria"""
        import random
        return {
            key: random.choice(values)
            for key, values in search_space.items()
        }
    
    def _grid_config(self, search_space: Dict[str, List[Any]], trial: int) -> Dict[str, Any]:
        """Configuración de grid"""
        total = 1
        for values in search_space.values():
            total *= len(values)
        
        if trial >= total:
            return self._random_config(search_space)
        
        config = {}
        remaining = trial
        
        for key, values in search_space.items():
            index = remaining % len(values)
            config[key] = values[index]
            remaining //= len(values)
        
        return config
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de tuning"""
        if not self.tuning_history:
            return {}
        
        scores = [h["score"] for h in self.tuning_history]
        
        return {
            "total_trials": len(self.tuning_history),
            "best_score": self.best_score,
            "best_config": self.best_config,
            "avg_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores)
        }




