"""
Backward compatibility re-export for templates.py

This file is deprecated. Use utils.templates.templates instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.templates.templates instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.templates.templates import *
