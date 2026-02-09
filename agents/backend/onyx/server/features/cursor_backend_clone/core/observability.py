"""
Backward compatibility re-export for observability.py

This file is deprecated. Use infrastructure.monitoring.observability instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.monitoring.observability instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.monitoring.observability import *
