"""
Backward compatibility re-export for serialization.py

This file is deprecated. Use utils.encoding.serialization instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.encoding.serialization instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.encoding.serialization import *
