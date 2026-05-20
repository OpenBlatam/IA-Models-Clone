"""
Backward compatibility re-export for profiling_utils.py

This file is deprecated. Use utils.performance.profiling_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.performance.profiling_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.performance.profiling_utils import *
