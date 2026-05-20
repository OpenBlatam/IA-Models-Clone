"""
Backward compatibility re-export for batch_processor.py

This file is deprecated. Use utils.async.batch_processor instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.async.batch_processor instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.async.batch_processor import *
