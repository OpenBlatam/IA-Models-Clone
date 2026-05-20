"""
Backward compatibility re-export for encoding_utils.py

This file is deprecated. Use utils.encoding.encoding_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.encoding.encoding_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.encoding.encoding_utils import *
