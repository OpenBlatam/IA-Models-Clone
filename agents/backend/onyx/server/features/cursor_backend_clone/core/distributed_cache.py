"""
Backward compatibility re-export for distributed_cache.py

This file is deprecated. Use infrastructure.caching.distributed_cache instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.caching.distributed_cache instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.caching.distributed_cache import *
