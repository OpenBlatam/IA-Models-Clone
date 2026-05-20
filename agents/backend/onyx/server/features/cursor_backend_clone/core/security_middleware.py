"""
Backward compatibility re-export for security_middleware.py

This file is deprecated. Use infrastructure.security.security_middleware instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.security.security_middleware instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.security.security_middleware import *
