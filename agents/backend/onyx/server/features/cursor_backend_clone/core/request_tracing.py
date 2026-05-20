"""
Backward compatibility re-export for request_tracing.py

This file is deprecated. Use utils.observability.request_tracing instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.observability.request_tracing instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.observability.request_tracing import *
