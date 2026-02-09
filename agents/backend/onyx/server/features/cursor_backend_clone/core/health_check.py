"""
Backward compatibility re-export for health_check.py

This file is deprecated. Use infrastructure.monitoring.health instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.monitoring.health instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.monitoring.health import *
