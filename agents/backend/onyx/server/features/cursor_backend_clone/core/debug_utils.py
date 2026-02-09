"""
Backward compatibility re-export for debug_utils.py

This file is deprecated. Use utils.debugging.debug_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.debugging.debug_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.debugging.debug_utils import *
