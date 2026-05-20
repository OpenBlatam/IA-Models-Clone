"""
Backward compatibility re-export for alerting.py

This file is deprecated. Use utils.alerts.alerting instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.alerts.alerting instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.alerts.alerting import *
