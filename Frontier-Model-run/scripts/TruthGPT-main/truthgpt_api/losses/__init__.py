"""
TruthGPT Losses Module
=====================

TensorFlow-like loss function implementations for TruthGPT.
"""

from .categorical_crossentropy import CategoricalCrossentropy, SparseCategoricalCrossentropy
from .binary_crossentropy import BinaryCrossentropy
from .mse import MeanSquaredError
from .mae import MeanAbsoluteError

__all__ = [
    'CategoricalCrossentropy', 'SparseCategoricalCrossentropy',
    'BinaryCrossentropy', 'MeanSquaredError', 'MeanAbsoluteError'
]









