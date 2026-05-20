"""
Backward compatibility re-export for event_bus.py

This file is deprecated. Use infrastructure.messaging.event_bus instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.messaging.event_bus instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.messaging.event_bus import *
