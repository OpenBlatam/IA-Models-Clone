"""
Backward compatibility re-export for data_validator.py

This file is deprecated. Use utils.data.data_validator instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.data.data_validator instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.data.data_validator import *
