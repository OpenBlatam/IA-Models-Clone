"""
Backward compatibility re-export for api_versioning.py

This file is deprecated. Use utils.api.api_versioning instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.api.api_versioning instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.api.api_versioning import *
