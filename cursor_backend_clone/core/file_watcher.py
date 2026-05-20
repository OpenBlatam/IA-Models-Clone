"""
Backward compatibility re-export for file_watcher.py

This file is deprecated. Use services.file_watcher instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use services.file_watcher instead.",
    DeprecationWarning,
    stacklevel=2
)

from .services.file_watcher import *
