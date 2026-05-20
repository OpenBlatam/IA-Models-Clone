"""
Backward compatibility re-export for user_rate_limiter.py

This file is deprecated. Use utils.validation.user_rate_limiter instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.validation.user_rate_limiter instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.validation.user_rate_limiter import *
