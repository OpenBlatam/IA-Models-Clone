"""
Backward compatibility re-export for dynamic_config.py

This file is deprecated. Use utils.config.dynamic_config instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.config.dynamic_config instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.config.dynamic_config import *
