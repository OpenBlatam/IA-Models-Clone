"""
Backward compatibility re-export for backup.py

This file is deprecated. Use infrastructure.persistence.backup instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.persistence.backup instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.persistence.backup import *
