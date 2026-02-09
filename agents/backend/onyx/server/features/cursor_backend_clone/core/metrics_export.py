"""
Backward compatibility re-export for metrics_export.py

This file is deprecated. Use utils.observability.metrics_export instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.observability.metrics_export instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.observability.metrics_export import *
