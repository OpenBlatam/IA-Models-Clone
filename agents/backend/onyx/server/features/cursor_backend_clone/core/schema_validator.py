"""
Backward compatibility re-export for schema_validator.py

This file is deprecated. Use utils.validation.schema_validator instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.validation.schema_validator instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.validation.schema_validator import *
