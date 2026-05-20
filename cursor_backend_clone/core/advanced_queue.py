"""
Backward compatibility re-export for advanced_queue.py

This file is deprecated. Use utils.async.advanced_queue instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.async.advanced_queue instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.async.advanced_queue import *
