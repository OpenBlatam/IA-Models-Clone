"""
TruthGPT Pruning Module
======================

TensorFlow-like pruning utilities for TruthGPT.
"""

from .magnitude_pruning import MagnitudePruning
from .structured_pruning import StructuredPruning
from .unstructured_pruning import UnstructuredPruning
from .base import BasePruning

__all__ = [
    'BasePruning', 'MagnitudePruning', 
    'StructuredPruning', 'UnstructuredPruning'
]


