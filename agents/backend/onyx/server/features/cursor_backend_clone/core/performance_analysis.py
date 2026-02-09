"""
Backward compatibility re-export for performance_analysis.py

This file is deprecated. Use utils.performance.performance_analysis instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.performance.performance_analysis instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.performance.performance_analysis import *
