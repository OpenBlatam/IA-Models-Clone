"""
Backward compatibility re-export for logger_config.py

This file is deprecated. Use utils.logging.logger_config instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.logging.logger_config instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.logging.logger_config import *
