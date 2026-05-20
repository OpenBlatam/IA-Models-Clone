"""
Backward compatibility re-export for mcp_client.py

This file is deprecated. Use mcp.client instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.client instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.client import *
