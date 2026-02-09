"""
Backward compatibility re-export for comparison_utils.py

This file is deprecated. Use utils.data.comparison_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.data.comparison_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.data.comparison_utils import *
