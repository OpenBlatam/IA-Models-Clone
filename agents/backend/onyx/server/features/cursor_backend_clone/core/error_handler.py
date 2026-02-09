"""
Backward compatibility re-export for error_handler.py

This file is deprecated. Use utils.error.error_handler instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.error.error_handler instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.error.error_handler import *
