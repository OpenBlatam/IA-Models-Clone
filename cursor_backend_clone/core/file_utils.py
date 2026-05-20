"""
Backward compatibility re-export for file_utils.py

This file is deprecated. Use utils.file.file_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.file.file_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.file.file_utils import *
