"""
Backward compatibility re-export for testing_utils.py

This file is deprecated. Use utils.testing.testing_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.testing.testing_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.testing.testing_utils import *
