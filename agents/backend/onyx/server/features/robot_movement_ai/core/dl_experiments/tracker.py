"""
Experiment Tracker
==================

Tracker base para experimentos.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TrackerType(Enum):
    """Tipo de tracker."""
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    NONE = "none"


class ExperimentTracker(ABC):
    """
    Tracker base para experimentos.
    
    Proporciona interfaz común para diferentes sistemas de tracking.
    """
    
    def __init__(self, experiment_name: str, project_name: Optional[str] = None):
        """
        Inicializar tracker.
        
        Args:
            experiment_name: Nombre del experimento
            project_name: Nombre del proyecto (opcional)
        """
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.initialized = False
    
    @abstractmethod
    def init(self, config: Dict[str, Any]):
        """
        Inicializar tracker.
        
        Args:
            config: Configuración del experimento
        """
        pass
    
    @abstractmethod
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Loggear métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor
            step: Paso (opcional)
        """
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Loggear múltiples métricas.
        
        Args:
            metrics: Diccionario de métricas
            step: Paso (opcional)
        """
        pass
    
    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Loggear hiperparámetros.
        
        Args:
            params: Diccionario de hiperparámetros
        """
        pass
    
    @abstractmethod
    def log_model(self, model, artifact_name: Optional[str] = None):
        """
        Loggear modelo.
        
        Args:
            model: Modelo a loggear
            artifact_name: Nombre del artifact (opcional)
        """
        pass
    
    @abstractmethod
    def finish(self):
        """Finalizar tracking."""
        pass


class DummyTracker(ExperimentTracker):
    """Tracker dummy que no hace nada (para testing)."""
    
    def init(self, config: Dict[str, Any]):
        """Inicializar (no-op)."""
        self.initialized = True
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Loggear métrica (no-op)."""
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Loggear métricas (no-op)."""
        pass
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Loggear hiperparámetros (no-op)."""
        pass
    
    def log_model(self, model, artifact_name: Optional[str] = None):
        """Loggear modelo (no-op)."""
        pass
    
    def finish(self):
        """Finalizar (no-op)."""
        pass


def create_tracker(
    tracker_type: TrackerType,
    experiment_name: str,
    project_name: Optional[str] = None
) -> ExperimentTracker:
    """
    Crear tracker según tipo.
    
    Args:
        tracker_type: Tipo de tracker
        experiment_name: Nombre del experimento
        project_name: Nombre del proyecto (opcional)
        
    Returns:
        Tracker instanciado
    """
    if tracker_type == TrackerType.TENSORBOARD:
        from .tensorboard_tracker import TensorBoardTracker
        return TensorBoardTracker(experiment_name, project_name)
    elif tracker_type == TrackerType.WANDB:
        from .wandb_tracker import WandBTracker
        return WandBTracker(experiment_name, project_name)
    elif tracker_type == TrackerType.NONE:
        return DummyTracker(experiment_name, project_name)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


