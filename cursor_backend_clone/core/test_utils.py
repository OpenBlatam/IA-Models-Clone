"""
Backward compatibility re-export for test_utils.py

This file is deprecated. Use utils.testing.test_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.testing.test_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.testing.test_utils import *
