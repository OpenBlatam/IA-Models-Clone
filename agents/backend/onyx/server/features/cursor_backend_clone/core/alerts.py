"""
Backward compatibility re-export for alerts.py

This file is deprecated. Use utils.alerts.alerts instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.alerts.alerts instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.alerts.alerts import *
