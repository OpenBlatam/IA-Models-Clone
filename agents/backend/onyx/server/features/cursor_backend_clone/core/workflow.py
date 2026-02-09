"""
Backward compatibility re-export for workflow.py

This file is deprecated. Use utils.async.workflow instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.async.workflow instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.async.workflow import *
