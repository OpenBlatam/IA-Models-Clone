"""
Hyperparameter Utils - Utilidades de Optimización de Hiperparámetros
======================================================================

Utilidades para optimización de hiperparámetros.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Intentar importar bibliotecas opcionales
try:
    import optuna
    _has_optuna = True
except ImportError:
    _has_optuna = False
    logger.warning("optuna not available, some hyperparameter optimization features will be limited")


@dataclass
class HyperparameterConfig:
    """
    Configuración de hiperparámetros.
    """
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    weight_decay: float = 0.01
    dropout: float = 0.1
    hidden_size: int = 128
    num_layers: int = 2
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'weight_decay': self.weight_decay,
            'dropout': self.dropout,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'warmup_steps': self.warmup_steps
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HyperparameterConfig':
        """Crear desde diccionario."""
        return cls(**config_dict)


class GridSearch:
    """
    Grid search para hiperparámetros.
    """
    
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        objective_fn: Callable,
        direction: str = "minimize"
    ):
        """
        Inicializar grid search.
        
        Args:
            param_grid: Grid de parámetros
            objective_fn: Función objetivo
            direction: Dirección ('minimize' o 'maximize')
        """
        self.param_grid = param_grid
        self.objective_fn = objective_fn
        self.direction = direction
        self.results = []
    
    def search(self) -> Dict[str, Any]:
        """
        Ejecutar grid search.
        
        Returns:
            Mejor configuración y resultados
        """
        from itertools import product
        
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        best_score = float('inf') if self.direction == "minimize" else float('-inf')
        best_params = None
        
        for params in product(*param_values):
            param_dict = dict(zip(param_names, params))
            score = self.objective_fn(param_dict)
            
            self.results.append({
                'params': param_dict,
                'score': score
            })
            
            if (self.direction == "minimize" and score < best_score) or \
               (self.direction == "maximize" and score > best_score):
                best_score = score
                best_params = param_dict
            
            logger.info(f"Params: {param_dict}, Score: {score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results
        }


class RandomSearch:
    """
    Random search para hiperparámetros.
    """
    
    def __init__(
        self,
        param_distributions: Dict[str, Callable],
        n_iter: int = 20,
        objective_fn: Callable = None,
        direction: str = "minimize"
    ):
        """
        Inicializar random search.
        
        Args:
            param_distributions: Distribuciones de parámetros
            n_iter: Número de iteraciones
            objective_fn: Función objetivo
            direction: Dirección
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.objective_fn = objective_fn
        self.direction = direction
        self.results = []
    
    def search(self) -> Dict[str, Any]:
        """
        Ejecutar random search.
        
        Returns:
            Mejor configuración y resultados
        """
        best_score = float('inf') if self.direction == "minimize" else float('-inf')
        best_params = None
        
        for i in range(self.n_iter):
            params = {}
            for param_name, distribution in self.param_distributions.items():
                params[param_name] = distribution()
            
            score = self.objective_fn(params)
            
            self.results.append({
                'params': params,
                'score': score
            })
            
            if (self.direction == "minimize" and score < best_score) or \
               (self.direction == "maximize" and score > best_score):
                best_score = score
                best_params = params
            
            logger.info(f"Iteration {i + 1}/{self.n_iter}, Params: {params}, Score: {score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results
        }


class OptunaOptimizer:
    """
    Optimizador usando Optuna.
    """
    
    def __init__(
        self,
        study_name: Optional[str] = None,
        direction: str = "minimize",
        storage: Optional[str] = None
    ):
        """
        Inicializar optimizador Optuna.
        
        Args:
            study_name: Nombre del estudio
            direction: Dirección
            storage: Storage para persistencia
        """
        if not _has_optuna:
            raise ImportError("optuna is required for OptunaOptimizer")
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True
        )
    
    def optimize(
        self,
        objective_fn: Callable,
        n_trials: int = 100,
        timeout: Optional[float] = None
    ) -> optuna.Study:
        """
        Optimizar hiperparámetros.
        
        Args:
            objective_fn: Función objetivo
            n_trials: Número de trials
            timeout: Timeout en segundos
            
        Returns:
            Estudio de Optuna
        """
        self.study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)
        return self.study
    
    def suggest_hyperparameters(self, trial: optuna.Trial, config: Dict) -> Dict[str, Any]:
        """
        Sugerir hiperparámetros desde trial.
        
        Args:
            trial: Trial de Optuna
            config: Configuración con sugerencias
            
        Returns:
            Diccionario de hiperparámetros
        """
        params = {}
        
        for param_name, param_config in config.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'float':
                if 'log' in param_config and param_config['log']:
                    params[param_name] = trial.suggest_loguniform(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                else:
                    params[param_name] = trial.suggest_uniform(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return params


class HyperparameterTuner:
    """
    Tuner completo de hiperparámetros.
    """
    
    def __init__(
        self,
        model_fn: Callable,
        train_fn: Callable,
        eval_fn: Callable,
        method: str = "random"
    ):
        """
        Inicializar tuner.
        
        Args:
            model_fn: Función para crear modelo
            train_fn: Función de entrenamiento
            eval_fn: Función de evaluación
            method: Método ('grid', 'random', 'optuna')
        """
        self.model_fn = model_fn
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.method = method
    
    def tune(
        self,
        param_space: Dict[str, Any],
        train_data: Any,
        val_data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Sintonizar hiperparámetros.
        
        Args:
            param_space: Espacio de parámetros
            train_data: Datos de entrenamiento
            val_data: Datos de validación
            **kwargs: Argumentos adicionales
            
        Returns:
            Mejor configuración
        """
        def objective(params):
            # Crear modelo
            model = self.model_fn(**params)
            
            # Entrenar
            self.train_fn(model, train_data, **params)
            
            # Evaluar
            score = self.eval_fn(model, val_data)
            
            return score
        
        if self.method == "grid":
            searcher = GridSearch(param_space, objective)
            return searcher.search()
        
        elif self.method == "random":
            n_iter = kwargs.get('n_iter', 20)
            param_distributions = {
                k: lambda v=v: np.random.choice(v) if isinstance(v, list) else v()
                for k, v in param_space.items()
            }
            searcher = RandomSearch(param_distributions, n_iter, objective)
            return searcher.search()
        
        elif self.method == "optuna":
            if not _has_optuna:
                raise ImportError("optuna is required for Optuna method")
            
            optimizer = OptunaOptimizer()
            study = optimizer.optimize(objective, n_trials=kwargs.get('n_trials', 100))
            
            return {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'study': study
            }
        
        else:
            raise ValueError(f"Unknown method: {self.method}")




