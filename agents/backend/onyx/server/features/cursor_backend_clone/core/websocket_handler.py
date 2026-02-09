"""
Backward compatibility re-export for websocket_handler.py

This file is deprecated. Use infrastructure.messaging.websocket instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use infrastructure.messaging.websocket instead.",
    DeprecationWarning,
    stacklevel=2
)

from .infrastructure.messaging.websocket import *
