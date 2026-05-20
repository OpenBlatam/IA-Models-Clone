"""
Backward compatibility re-export for http_client.py

This file is deprecated. Use utils.network.http_client instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use utils.network.http_client instead.",
    DeprecationWarning,
    stacklevel=2
)

from .utils.network.http_client import *
