"""
Initialization Module
====================

Model weight initialization strategies and utilities.
"""

from .weight_initializer import WeightInitializer
from .initialization_strategies import InitializationStrategies

__all__ = [
    "WeightInitializer",
    "InitializationStrategies",
]


