"""
Backward compatibility re-export for plugins.py

This file is deprecated. Use infrastructure.plugins.plugins instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.plugins.plugins instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.plugins.plugins import *
