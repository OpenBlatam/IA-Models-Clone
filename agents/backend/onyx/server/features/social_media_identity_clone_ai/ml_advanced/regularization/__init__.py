"""Regularization module."""

from .advanced_regularization import (
    DropBlock,
    SpectralNormalization,
    LabelSmoothingRegularization,
    MixUp
)

__all__ = [
    "DropBlock",
    "SpectralNormalization",
    "LabelSmoothingRegularization",
    "MixUp",
]




