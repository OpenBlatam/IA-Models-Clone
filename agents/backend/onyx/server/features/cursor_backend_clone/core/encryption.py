"""
Backward compatibility re-export for encryption.py

This file is deprecated. Use utils.security.encryption instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.security.encryption instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.security.encryption import *
