"""
Backward compatibility re-export for logging_config.py

This file is deprecated. Use utils.logging.logging_config instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.logging.logging_config instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.logging.logging_config import *
