"""
Backward compatibility re-export for security_audit.py

This file is deprecated. Use infrastructure.security.security_audit instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.security.security_audit instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.security.security_audit import *
