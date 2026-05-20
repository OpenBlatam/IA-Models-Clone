"""
Backward compatibility re-export for decorator_utils.py

This file is deprecated. Use utils.decorators.decorator_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.decorators.decorator_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.decorators.decorator_utils import *
