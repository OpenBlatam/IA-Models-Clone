"""
ML Module - Módulo de Machine Learning
=======================================

Módulo modular para componentes de Machine Learning y Deep Learning.
"""

from .models import BaseModel, CodeCompletionModel, CodeExplanationModel
from .data import CodeDataset, DataCollator, DataLoaderFactory
from .training import Trainer, TrainingConfig, TrainingCallback
from .evaluation import Evaluator, EvaluationMetrics
from .config import MLConfig, load_config

__all__ = [
    "BaseModel",
    "CodeCompletionModel",
    "CodeExplanationModel",
    "CodeDataset",
    "DataCollator",
    "DataLoaderFactory",
    "Trainer",
    "TrainingConfig",
    "TrainingCallback",
    "Evaluator",
    "EvaluationMetrics",
    "MLConfig",
    "load_config",
]


