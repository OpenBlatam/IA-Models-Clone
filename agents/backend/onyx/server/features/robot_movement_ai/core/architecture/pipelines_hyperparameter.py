"""
Hyperparameter Tuning Module
==============================

Sistema profesional para búsqueda de hiperparámetros.
Incluye grid search, random search, y Bayesian optimization.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import random

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not available. Bayesian optimization disabled.")

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Espacio de búsqueda de hiperparámetros."""
    learning_rate: Tuple[float, float] = (1e-5, 1e-2)
    batch_size: List[int] = None
    num_epochs: int = 100
    weight_decay: Tuple[float, float] = (1e-6, 1e-2)
    dropout: Tuple[float, float] = (0.0, 0.5)
    hidden_size: List[int] = None
    
    def __post_init__(self):
        """Inicializar valores por defecto."""
        if self.batch_size is None:
            self.batch_size = [16, 32, 64, 128]
        if self.hidden_size is None:
            self.hidden_size = [64, 128, 256, 512]


class HyperparameterTuner:
    """
    Tuner de hiperparámetros profesional.
    
    Soporta:
    - Grid Search
    - Random Search
    - Bayesian Optimization (si scikit-optimize está disponible)
    """
    
    def __init__(
        self,
        model_fn: Callable,
        train_fn: Callable,
        eval_fn: Callable,
        space: HyperparameterSpace,
        method: str = "random"
    ):
        """
        Inicializar tuner.
        
        Args:
            model_fn: Función que crea modelo dado hiperparámetros
            train_fn: Función de entrenamiento
            eval_fn: Función de evaluación (retorna score)
            space: Espacio de búsqueda
            method: Método ("grid", "random", "bayesian")
        """
        self.model_fn = model_fn
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.space = space
        self.method = method
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf')
    
    def grid_search(
        self,
        n_trials: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Grid search exhaustivo.
        
        Args:
            n_trials: Número máximo de trials (None para todos)
            verbose: Mostrar progreso
            
        Returns:
            Mejores hiperparámetros
        """
        # Generar todas las combinaciones
        param_grid = self._generate_param_grid()
        
        if n_trials:
            param_grid = param_grid[:n_trials]
        
        logger.info(f"Grid search: {len(param_grid)} combinations")
        
        for i, params in enumerate(param_grid):
            if verbose:
                logger.info(f"Trial {i+1}/{len(param_grid)}: {params}")
            
            score = self._evaluate_params(params)
            self.results.append({"params": params, "score": score})
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                if verbose:
                    logger.info(f"New best score: {score:.4f}")
        
        return self.best_params
    
    def random_search(
        self,
        n_trials: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Random search.
        
        Args:
            n_trials: Número de trials
            verbose: Mostrar progreso
            
        Returns:
            Mejores hiperparámetros
        """
        logger.info(f"Random search: {n_trials} trials")
        
        for i in range(n_trials):
            params = self._sample_random_params()
            
            if verbose:
                logger.info(f"Trial {i+1}/{n_trials}: {params}")
            
            score = self._evaluate_params(params)
            self.results.append({"params": params, "score": score})
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                if verbose:
                    logger.info(f"New best score: {score:.4f}")
        
        return self.best_params
    
    def bayesian_optimization(
        self,
        n_calls: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Bayesian optimization usando scikit-optimize.
        
        Args:
            n_calls: Número de llamadas
            verbose: Mostrar progreso
            
        Returns:
            Mejores hiperparámetros
        """
        if not SKOPT_AVAILABLE:
            logger.warning("scikit-optimize not available, falling back to random search")
            return self.random_search(n_trials=n_calls, verbose=verbose)
        
        # Definir espacio de búsqueda
        dimensions = [
            Real(self.space.learning_rate[0], self.space.learning_rate[1], prior='log-uniform', name='learning_rate'),
            Categorical(self.space.batch_size, name='batch_size'),
            Real(self.space.weight_decay[0], self.space.weight_decay[1], prior='log-uniform', name='weight_decay'),
            Real(self.space.dropout[0], self.space.dropout[1], name='dropout'),
            Categorical(self.space.hidden_size, name='hidden_size'),
        ]
        
        def objective(params):
            """Función objetivo para optimización."""
            param_dict = {
                'learning_rate': params[0],
                'batch_size': params[1],
                'weight_decay': params[2],
                'dropout': params[3],
                'hidden_size': params[4]
            }
            
            score = self._evaluate_params(param_dict)
            return -score  # Minimizar (negativo porque queremos maximizar)
        
        logger.info(f"Bayesian optimization: {n_calls} calls")
        
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=42,
            verbose=verbose
        )
        
        # Convertir resultado
        self.best_params = {
            'learning_rate': result.x[0],
            'batch_size': result.x[1],
            'weight_decay': result.x[2],
            'dropout': result.x[3],
            'hidden_size': result.x[4]
        }
        self.best_score = -result.fun
        
        return self.best_params
    
    def _generate_param_grid(self) -> List[Dict[str, Any]]:
        """Generar grid de parámetros."""
        import itertools
        
        lr_values = np.logspace(
            np.log10(self.space.learning_rate[0]),
            np.log10(self.space.learning_rate[1]),
            num=5
        )
        
        combinations = list(itertools.product(
            lr_values,
            self.space.batch_size,
            self.space.hidden_size
        ))
        
        return [
            {
                'learning_rate': lr,
                'batch_size': bs,
                'hidden_size': hs,
                'weight_decay': 1e-4,
                'dropout': 0.2
            }
            for lr, bs, hs in combinations
        ]
    
    def _sample_random_params(self) -> Dict[str, Any]:
        """Muestrear parámetros aleatorios."""
        return {
            'learning_rate': np.random.uniform(
                self.space.learning_rate[0],
                self.space.learning_rate[1]
            ),
            'batch_size': random.choice(self.space.batch_size),
            'weight_decay': np.random.uniform(
                self.space.weight_decay[0],
                self.space.weight_decay[1]
            ),
            'dropout': np.random.uniform(
                self.space.dropout[0],
                self.space.dropout[1]
            ),
            'hidden_size': random.choice(self.space.hidden_size)
        }
    
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """
        Evaluar conjunto de hiperparámetros.
        
        Args:
            params: Diccionario de hiperparámetros
            
        Returns:
            Score de evaluación
        """
        try:
            # Crear modelo
            model = self.model_fn(params)
            
            # Entrenar
            self.train_fn(model, params)
            
            # Evaluar
            score = self.eval_fn(model)
            
            return score
        except Exception as e:
            logger.error(f"Error evaluating params {params}: {e}")
            return float('-inf')
    
    def tune(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Ejecutar búsqueda de hiperparámetros.
        
        Args:
            n_trials: Número de trials
            
        Returns:
            Mejores hiperparámetros
        """
        if self.method == "grid":
            return self.grid_search(n_trials=n_trials)
        elif self.method == "random":
            return self.random_search(n_trials=n_trials)
        elif self.method == "bayesian":
            return self.bayesian_optimization(n_calls=n_trials)
        else:
            raise ValueError(f"Unknown method: {self.method}")

