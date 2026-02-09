"""
Backward compatibility re-export for path_utils.py

This file is deprecated. Use utils.file.path_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.file.path_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.file.path_utils import *
