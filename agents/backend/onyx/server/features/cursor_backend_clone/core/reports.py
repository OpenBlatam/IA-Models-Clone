"""
Backward compatibility re-export for reports.py

This file is deprecated. Use utils.api.reports instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.api.reports instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.api.reports import *
