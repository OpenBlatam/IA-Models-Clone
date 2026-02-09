"""Interfaces for ML components."""

from .prediction_interface import IPredictionService
from .training_interface import ITrainingService
from .evaluation_interface import IEvaluationService

__all__ = [
    "IPredictionService",
    "ITrainingService",
    "IEvaluationService",
]




