"""
Backward compatibility re-export for time_utils.py

This file is deprecated. Use utils.time.time_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.time.time_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.time.time_utils import *
