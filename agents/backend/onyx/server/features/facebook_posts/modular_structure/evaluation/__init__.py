"""
📊 Evaluation Module

This module contains all evaluation-related classes and implementations.
Separated from models and training for better modularity.
"""

from .base_evaluator import BaseEvaluator
from .classification_evaluator import ClassificationEvaluator
from .regression_evaluator import RegressionEvaluator
from .generation_evaluator import GenerationEvaluator
from .metrics_calculator import MetricsCalculator
from .results_visualizer import ResultsVisualizer
from .model_comparison import ModelComparison

__all__ = [
    "BaseEvaluator",
    "ClassificationEvaluator",
    "RegressionEvaluator", 
    "GenerationEvaluator",
    "MetricsCalculator",
    "ResultsVisualizer",
    "ModelComparison"
]






