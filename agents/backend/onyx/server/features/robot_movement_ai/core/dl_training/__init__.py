"""
Training Module
===============

Módulo modular para entrenamiento de modelos.
"""

from .trainer import ModelTrainer, TrainingConfig, TrainingStrategy
from .callbacks import Callback, EarlyStoppingCallback, LearningRateSchedulerCallback
from .optimizers import create_optimizer, OptimizerType
from .schedulers import create_scheduler, SchedulerType

__all__ = [
    "ModelTrainer",
    "TrainingConfig",
    "TrainingStrategy",
    "Callback",
    "EarlyStoppingCallback",
    "LearningRateSchedulerCallback",
    "create_optimizer",
    "OptimizerType",
    "create_scheduler",
    "SchedulerType",
]




