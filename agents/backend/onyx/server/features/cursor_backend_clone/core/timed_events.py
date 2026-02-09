"""
Backward compatibility re-export for timed_events.py

This file is deprecated. Use infrastructure.scheduling.timed_events instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.scheduling.timed_events instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.scheduling.timed_events import *
