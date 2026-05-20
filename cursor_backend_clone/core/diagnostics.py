"""
Backward compatibility re-export for diagnostics.py

This file is deprecated. Use infrastructure.monitoring.diagnostics instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.monitoring.diagnostics instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.monitoring.diagnostics import *
