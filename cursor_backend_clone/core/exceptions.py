"""
Backward compatibility re-export for exceptions.py

This file is deprecated. Use domain.exceptions instead.
"""
import warnings

warnings.warn(
    "exceptions.py is deprecated. Use domain.exceptions instead.",
    DeprecationWarning,
    stacklevel=2
)

from .domain.exceptions import *
