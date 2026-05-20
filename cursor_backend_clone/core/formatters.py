"""
Backward compatibility re-export for formatters.py

This file is deprecated. Use utils.text.formatters instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.text.formatters instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.text.formatters import *
