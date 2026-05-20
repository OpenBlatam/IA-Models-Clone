"""
Backward compatibility re-export for migrations.py

This file is deprecated. Use utils.distributed.migrations instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.distributed.migrations instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.distributed.migrations import *
