"""
Backward compatibility re-export for context_utils.py

This file is deprecated. Use utils.context.context_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.context.context_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.context.context_utils import *
