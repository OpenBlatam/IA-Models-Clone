"""
Backward compatibility re-export for throttle.py

This file is deprecated. Use utils.performance.throttle instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.performance.throttle instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.performance.throttle import *
