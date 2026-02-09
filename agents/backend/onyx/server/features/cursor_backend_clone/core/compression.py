"""
Backward compatibility re-export for compression.py

This file is deprecated. Use utils.encoding.compression instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.encoding.compression instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.encoding.compression import *
