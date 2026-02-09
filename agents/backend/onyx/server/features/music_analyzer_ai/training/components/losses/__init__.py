"""
Loss Functions Module

Modular loss functions for different tasks.
"""

from .classification import (
    ClassificationLoss,
    FocalLoss,
    LabelSmoothingLoss
)
from .regression import (
    RegressionLoss,
    SmoothL1Loss
)

# Re-export for backward compatibility
__all__ = [
    "ClassificationLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "RegressionLoss",
    "SmoothL1Loss",
]



