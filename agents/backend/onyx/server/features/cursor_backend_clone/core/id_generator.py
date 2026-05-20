"""
Backward compatibility re-export for id_generator.py

This file is deprecated. Use utils.id.id_generator instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.id.id_generator instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.id.id_generator import *
