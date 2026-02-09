"""
Backward compatibility re-export for config_manager.py

This file is deprecated. Use utils.config.config_manager instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.config.config_manager instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.config.config_manager import *
