"""
Backward compatibility re-export for circuit_breaker.py

This file is deprecated. Use utils.retry.circuit_breaker instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.retry.circuit_breaker instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.retry.circuit_breaker import *
