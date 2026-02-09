"""
Backward compatibility re-export for data_transform.py

This file is deprecated. Use utils.data.data_transform instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.data.data_transform instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.data.data_transform import *
