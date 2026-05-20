"""
Backward compatibility re-export for text_utils.py

This file is deprecated. Use utils.text.text_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.text.text_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.text.text_utils import *
