"""
Backward compatibility re-export for api_docs.py

This file is deprecated. Use utils.api.api_docs instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.api.api_docs instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.api.api_docs import *
