"""
Backward compatibility re-export for performance.py

This file is deprecated. Use utils.performance.performance instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.performance.performance instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.performance.performance import *
