"""
Backward compatibility re-export for auth.py

This file is deprecated. Use infrastructure.security.auth instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.security.auth instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.security.auth import *
