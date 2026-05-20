"""
Backward compatibility re-export for statistics.py

This file is deprecated. Use utils.data.statistics instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.data.statistics instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.data.statistics import *
