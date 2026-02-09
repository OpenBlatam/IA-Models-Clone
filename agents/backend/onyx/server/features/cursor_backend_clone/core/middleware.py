"""
Backward compatibility re-export for middleware.py

This file is deprecated. Use utils.middleware.middleware instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.middleware.middleware instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.middleware.middleware import *
