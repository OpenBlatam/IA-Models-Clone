"""
Backward compatibility re-export for async_utils.py

This file is deprecated. Use utils.async.async_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.async.async_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.async.async_utils import *
