"""
Backward compatibility re-export for cache.py

This file is deprecated. Use infrastructure.caching.cache instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.caching.cache instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.caching.cache import *
