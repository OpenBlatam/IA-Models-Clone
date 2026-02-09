"""
AutoML Utils - Utilidades de AutoML
====================================

Utilidades para AutoML y automatización de pipelines ML.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:
    """Configuración de AutoML."""
    max_trials: int = 100
    timeout: Optional[float] = None
    metric: str = "accuracy"
    direction: str = "maximize"
    search_space: Dict[str, List[Any]] = None


class AutoMLPipeline:
    """
    Pipeline automático de ML.
    """
    
    def __init__(
        self,
        model_builder: Callable,
        train_fn: Callable,
        eval_fn: Callable,
        config: Optional[AutoMLConfig] = None
    ):
        """
        Inicializar pipeline AutoML.
        
        Args:
            model_builder: Función para construir modelo
            train_fn: Función de entrenamiento
            eval_fn: Función de evaluación
            config: Configuración (opcional)
        """
        self.model_builder = model_builder
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.config = config or AutoMLConfig()
        self.best_model = None
        self.best_score = float('-inf') if self.config.direction == "maximize" else float('inf')
        self.best_config = None
        self.trials = []
    
    def search(
        self,
        train_data: Any,
        val_data: Any,
        search_space: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Any]:
        """
        Buscar mejor configuración.
        
        Args:
            train_data: Datos de entrenamiento
            val_data: Datos de validación
            search_space: Espacio de búsqueda (opcional)
            
        Returns:
            Mejor configuración y modelo
        """
        search_space = search_space or self.config.search_space
        if search_space is None:
            raise ValueError("search_space must be provided")
        
        import random
        import time
        
        start_time = time.time()
        
        for trial in range(self.config.max_trials):
            # Verificar timeout
            if self.config.timeout and (time.time() - start_time) > self.config.timeout:
                logger.info(f"Timeout reached after {trial} trials")
                break
            
            # Generar configuración
            config = self._sample_config(search_space)
            
            # Construir y entrenar modelo
            try:
                model = self.model_builder(**config)
                self.train_fn(model, train_data, **config)
                
                # Evaluar
                score = self.eval_fn(model, val_data)
                
                self.trials.append({
                    'config': config,
                    'score': score
                })
                
                # Actualizar mejor
                is_better = (
                    (self.config.direction == "maximize" and score > self.best_score) or
                    (self.config.direction == "minimize" and score < self.best_score)
                )
                
                if is_better:
                    self.best_score = score
                    self.best_config = config
                    self.best_model = model
                
                logger.info(f"Trial {trial + 1}/{self.config.max_trials}, Score: {score:.4f}, Best: {self.best_score:.4f}")
            
            except Exception as e:
                logger.warning(f"Trial {trial + 1} failed: {e}")
                continue
        
        return {
            'best_model': self.best_model,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'trials': self.trials
        }
    
    def _sample_config(self, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Muestrear configuración.
        
        Args:
            search_space: Espacio de búsqueda
            
        Returns:
            Configuración
        """
        import random
        config = {}
        for key, values in search_space.items():
            config[key] = random.choice(values)
        return config


class AutoFeatureEngineering:
    """
    Feature engineering automático.
    """
    
    def __init__(self):
        """Inicializar feature engineering automático."""
        self.transformations = []
    
    def auto_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transformación automática.
        
        Args:
            X: Features
            y: Targets (opcional)
            
        Returns:
            Features transformadas
        """
        result = X
        
        # Escalado
        from .feature_engineering_utils import FeatureScaler
        scaler = FeatureScaler("standard")
        result = scaler.fit_transform(result)
        
        # Selección de features (si hay targets)
        if y is not None and result.shape[1] > 50:
            from .feature_engineering_utils import FeatureSelector
            selector = FeatureSelector("kbest", k=50)
            result = selector.fit_transform(result, y)
        
        return result


class AutoHyperparameterTuning:
    """
    Tuning automático de hiperparámetros.
    """
    
    def __init__(
        self,
        model_class: type,
        search_space: Dict[str, List[Any]],
        objective_fn: Callable
    ):
        """
        Inicializar tuning.
        
        Args:
            model_class: Clase del modelo
            search_space: Espacio de búsqueda
            objective_fn: Función objetivo
        """
        self.model_class = model_class
        self.search_space = search_space
        self.objective_fn = objective_fn
    
    def tune(
        self,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Sintonizar hiperparámetros.
        
        Args:
            n_trials: Número de trials
            
        Returns:
            Mejor configuración
        """
        from .hyperparameter_utils import RandomSearch
        
        searcher = RandomSearch(
            self.search_space,
            n_trials=n_trials,
            objective_fn=self.objective_fn
        )
        
        return searcher.search()


def create_automl_pipeline(
    task_type: str = "classification",
    search_space: Optional[Dict[str, List[Any]]] = None
) -> AutoMLPipeline:
    """
    Crear pipeline AutoML.
    
    Args:
        task_type: Tipo de tarea
        search_space: Espacio de búsqueda (opcional)
        
    Returns:
        Pipeline AutoML
    """
    # Implementación simplificada
    def model_builder(**kwargs):
        from .model_utils import ModelBuilder
        return ModelBuilder.create_mlp(
            input_size=kwargs.get('input_size', 784),
            hidden_sizes=kwargs.get('hidden_sizes', [128, 64]),
            output_size=kwargs.get('output_size', 10)
        )
    
    def train_fn(model, data, **kwargs):
        # Placeholder
        pass
    
    def eval_fn(model, data):
        # Placeholder
        return 0.0
    
    return AutoMLPipeline(model_builder, train_fn, eval_fn)




