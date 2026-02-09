"""
Routing Training Package
========================

Módulos para entrenamiento de modelos de enrutamiento.
"""

from .trainer import RouteTrainer, TrainingConfig
from .callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from .metrics import MetricsCalculator, RouteMetrics
from .distributed_training import DistributedTrainer, GradientAccumulator
from .hyperparameter_optimization import HyperparameterOptimizer, create_objective_function
from .fast_trainer import FastRouteTrainer

__all__ = [
    "RouteTrainer",
    "FastRouteTrainer",
    "TrainingConfig",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "MetricsCalculator",
    "RouteMetrics",
    "DistributedTrainer",
    "GradientAccumulator",
    "HyperparameterOptimizer",
    "create_objective_function"
]

# Imports condicionales para callbacks avanzados
try:
    from .advanced_callbacks import (
        LearningRateFinder,
        GradientMonitor,
        ModelEMA,
        ProfilerCallback
    )
    __all__.extend([
        "LearningRateFinder",
        "GradientMonitor",
        "ModelEMA",
        "ProfilerCallback"
    ])
except ImportError:
    pass

