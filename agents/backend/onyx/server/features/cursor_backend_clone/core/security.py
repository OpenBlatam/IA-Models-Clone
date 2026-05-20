"""
Backward compatibility re-export for security.py

This file is deprecated. Use infrastructure.security.security instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.security.security instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.security.security import *
