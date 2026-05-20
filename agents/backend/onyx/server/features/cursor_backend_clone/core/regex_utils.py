"""
Backward compatibility re-export for regex_utils.py

This file is deprecated. Use utils.regex.regex_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.regex.regex_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.regex.regex_utils import *
