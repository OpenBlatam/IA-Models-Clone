"""
Backward compatibility re-export for persistent_service.py

This file is deprecated. Use services.persistent_service instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use services.persistent_service instead.",
    DeprecationWarning,
    stacklevel=2
)

from .services.persistent_service import *
