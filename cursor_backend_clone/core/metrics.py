"""
Backward compatibility re-export for metrics.py

This file is deprecated. Use infrastructure.monitoring.metrics instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.monitoring.metrics instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.monitoring.metrics import *
