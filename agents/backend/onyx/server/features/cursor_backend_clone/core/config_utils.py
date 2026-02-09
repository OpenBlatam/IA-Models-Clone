"""
Backward compatibility re-export for config_utils.py

This file is deprecated. Use utils.config.config_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.config.config_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.config.config_utils import *
