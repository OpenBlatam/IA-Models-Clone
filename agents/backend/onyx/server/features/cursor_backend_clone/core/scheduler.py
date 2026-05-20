"""
Backward compatibility re-export for scheduler.py

This file is deprecated. Use infrastructure.scheduling.scheduler instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.scheduling.scheduler instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.scheduling.scheduler import *
