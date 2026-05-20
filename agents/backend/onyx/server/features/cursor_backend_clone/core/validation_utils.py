"""
Backward compatibility re-export for validation_utils.py

This file is deprecated. Use utils.validation.validation_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.validation.validation_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.validation.validation_utils import *
