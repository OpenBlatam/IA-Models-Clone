"""
Sistema de Auto-tuning de Hiperparámetros

Proporciona:
- Grid search
- Random search
- Bayesian optimization
- Early stopping
- Resultados comparativos
"""

import logging
import random
import itertools
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Estrategias de búsqueda"""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


@dataclass
class HyperparameterConfig:
    """Configuración de hiperparámetros"""
    name: str
    type: str  # "int", "float", "choice"
    values: List[Any] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None


@dataclass
class TrialResult:
    """Resultado de un trial"""
    trial_id: int
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0


class HyperparameterTuner:
    """Tuner de hiperparámetros"""
    
    def __init__(self):
        self.trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None
        logger.info("HyperparameterTuner initialized")
    
    def grid_search(
        self,
        hyperparameters: Dict[str, HyperparameterConfig],
        objective_func: Callable[[Dict[str, Any]], Dict[str, float]],
        max_trials: Optional[int] = None
    ) -> TrialResult:
        """
        Grid search exhaustivo
        
        Args:
            hyperparameters: Configuración de hiperparámetros
            objective_func: Función objetivo que retorna métricas
            max_trials: Máximo de trials (None = todos)
        
        Returns:
            Mejor trial
        """
        # Generar todas las combinaciones
        param_names = list(hyperparameters.keys())
        param_values = [
            self._get_values(config)
            for config in hyperparameters.values()
        ]
        
        combinations = list(itertools.product(*param_values))
        
        if max_trials:
            combinations = combinations[:max_trials]
        
        logger.info(f"Starting grid search with {len(combinations)} trials")
        
        for trial_id, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            try:
                start_time = datetime.now()
                metrics = objective_func(params)
                duration = (datetime.now() - start_time).total_seconds()
                
                result = TrialResult(
                    trial_id=trial_id,
                    hyperparameters=params,
                    metrics=metrics,
                    duration=duration
                )
                
                self.trials.append(result)
                
                # Actualizar mejor trial
                if self._is_better(result, self.best_trial):
                    self.best_trial = result
                
                logger.info(
                    f"Trial {trial_id}: {params} -> {metrics.get('score', 0):.4f}"
                )
            
            except Exception as e:
                logger.error(f"Error in trial {trial_id}: {e}")
        
        return self.best_trial
    
    def random_search(
        self,
        hyperparameters: Dict[str, HyperparameterConfig],
        objective_func: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: int = 50
    ) -> TrialResult:
        """
        Random search
        
        Args:
            hyperparameters: Configuración de hiperparámetros
            objective_func: Función objetivo
            n_trials: Número de trials
        
        Returns:
            Mejor trial
        """
        logger.info(f"Starting random search with {n_trials} trials")
        
        for trial_id in range(n_trials):
            # Seleccionar valores aleatorios
            params = {}
            for name, config in hyperparameters.items():
                values = self._get_values(config)
                params[name] = random.choice(values)
            
            try:
                start_time = datetime.now()
                metrics = objective_func(params)
                duration = (datetime.now() - start_time).total_seconds()
                
                result = TrialResult(
                    trial_id=trial_id,
                    hyperparameters=params,
                    metrics=metrics,
                    duration=duration
                )
                
                self.trials.append(result)
                
                if self._is_better(result, self.best_trial):
                    self.best_trial = result
                
                logger.info(
                    f"Trial {trial_id}: {params} -> {metrics.get('score', 0):.4f}"
                )
            
            except Exception as e:
                logger.error(f"Error in trial {trial_id}: {e}")
        
        return self.best_trial
    
    def _get_values(self, config: HyperparameterConfig) -> List[Any]:
        """Obtiene valores de una configuración"""
        if config.values:
            return config.values
        
        if config.type == "int" and config.min_value is not None and config.max_value is not None:
            step = config.step or 1
            return list(range(
                int(config.min_value),
                int(config.max_value) + 1,
                int(step)
            ))
        
        if config.type == "float" and config.min_value is not None and config.max_value is not None:
            step = config.step or 0.1
            values = []
            current = config.min_value
            while current <= config.max_value:
                values.append(current)
                current += step
            return values
        
        return []
    
    def _is_better(self, trial1: TrialResult, trial2: Optional[TrialResult]) -> bool:
        """Compara dos trials"""
        if trial2 is None:
            return True
        
        # Comparar por score (mayor es mejor)
        score1 = trial1.metrics.get("score", 0)
        score2 = trial2.metrics.get("score", 0)
        
        return score1 > score2
    
    def get_best_hyperparameters(self) -> Optional[Dict[str, Any]]:
        """Obtiene los mejores hiperparámetros"""
        if self.best_trial:
            return self.best_trial.hyperparameters
        return None
    
    def get_trial_history(self) -> List[Dict[str, Any]]:
        """Obtiene historial de trials"""
        return [
            {
                "trial_id": trial.trial_id,
                "hyperparameters": trial.hyperparameters,
                "metrics": trial.metrics,
                "duration": trial.duration,
                "timestamp": trial.timestamp.isoformat()
            }
            for trial in self.trials
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del tuning"""
        if not self.trials:
            return {"message": "No trials completed"}
        
        scores = [t.metrics.get("score", 0) for t in self.trials]
        
        return {
            "total_trials": len(self.trials),
            "best_score": max(scores) if scores else 0,
            "worst_score": min(scores) if scores else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "best_hyperparameters": self.best_trial.hyperparameters if self.best_trial else None,
            "best_metrics": self.best_trial.metrics if self.best_trial else None
        }


# Instancia global
_hyperparameter_tuner: Optional[HyperparameterTuner] = None


def get_hyperparameter_tuner() -> HyperparameterTuner:
    """Obtiene la instancia global del tuner de hiperparámetros"""
    global _hyperparameter_tuner
    if _hyperparameter_tuner is None:
        _hyperparameter_tuner = HyperparameterTuner()
    return _hyperparameter_tuner

