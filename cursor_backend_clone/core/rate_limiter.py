"""
Backward compatibility re-export for rate_limiter.py

This file is deprecated. Use utils.rate_limiting.rate_limiter instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.rate_limiting.rate_limiter instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.rate_limiting.rate_limiter import *
