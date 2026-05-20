"""
Backward compatibility re-export for collection_utils.py

This file is deprecated. Use utils.data.collection_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.data.collection_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.data.collection_utils import *
