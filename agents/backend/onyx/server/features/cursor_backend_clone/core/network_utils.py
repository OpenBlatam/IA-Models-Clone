"""
Backward compatibility re-export for network_utils.py

This file is deprecated. Use utils.network.network_utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.network.network_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.network.network_utils import *
