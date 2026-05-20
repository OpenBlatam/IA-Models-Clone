"""
Backward compatibility re-export for retry_strategy.py

This file is deprecated. Use utils.retry.retry_strategy instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.retry.retry_strategy instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.retry.retry_strategy import *
