"""
Backward compatibility re-export for notifications.py

This file is deprecated. Use infrastructure.messaging.notifications instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.messaging.notifications instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.messaging.notifications import *
