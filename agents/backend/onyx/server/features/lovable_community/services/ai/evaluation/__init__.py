"""
Evaluation Module

Handles model evaluation organized into sub-modules:
- metrics: Evaluation metrics (accuracy, F1, etc.)
- evaluators: Model evaluators and cross-validation
"""

from .metrics import (
    ClassificationMetrics
)

from .evaluators import (
    ModelEvaluator,
    cross_validate
)

__all__ = [
    "ClassificationMetrics",
    "ModelEvaluator",
    "cross_validate",
]

