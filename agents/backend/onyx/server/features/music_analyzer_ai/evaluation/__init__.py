"""
Evaluation module
"""

# Import metrics (backward compatibility)
try:
    from .metrics import (
        ClassificationMetrics,
        RegressionMetrics,
        ModelEvaluator,
        CrossValidator
    )
except ImportError:
    # Fallback to submodule
    from .metrics.classification_metrics import ClassificationMetrics
    from .metrics.regression_metrics import RegressionMetrics
    # Try to import others from modular_metrics
    try:
        from .modular_metrics import ModelEvaluator, CrossValidator
    except ImportError:
        ModelEvaluator = None
        CrossValidator = None

__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
    "ModelEvaluator",
    "CrossValidator",
]

