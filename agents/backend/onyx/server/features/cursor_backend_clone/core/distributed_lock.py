"""
Backward compatibility re-export for distributed_lock.py

This file is deprecated. Use utils.distributed.distributed_lock instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.distributed.distributed_lock instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.distributed.distributed_lock import *
