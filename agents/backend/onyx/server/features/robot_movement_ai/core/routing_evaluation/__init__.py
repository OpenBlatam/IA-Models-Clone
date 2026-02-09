"""
Routing Evaluation Package
===========================

Módulos para evaluación avanzada de modelos.
"""

from .evaluator import ModelEvaluator, EvaluationConfig, EvaluationMetrics
from .cross_validation import CrossValidator, KFoldCrossValidator
from .model_comparison import ModelComparator, compare_models
from .visualization import EvaluationVisualizer, plot_training_curves, plot_confusion_matrix

__all__ = [
    "ModelEvaluator",
    "EvaluationConfig",
    "EvaluationMetrics",
    "CrossValidator",
    "KFoldCrossValidator",
    "ModelComparator",
    "compare_models",
    "EvaluationVisualizer",
    "plot_training_curves",
    "plot_confusion_matrix"
]

