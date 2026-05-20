"""
Backward compatibility re-export for search_utils.py

This file is deprecated. Use utils.search.search_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.search.search_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.search.search_utils import *
