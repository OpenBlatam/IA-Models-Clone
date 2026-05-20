"""
Evaluation Metrics Module

Modular metrics for different tasks.
"""

from .classification_metrics import ClassificationMetrics
from .regression_metrics import RegressionMetrics

# Re-export for backward compatibility
try:
    from ..modular_metrics import (
        MultiTaskMetrics
    )
except ImportError:
    MultiTaskMetrics = None

__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
    "MultiTaskMetrics",
]

