"""
Backward compatibility re-export for mcp_auth.py

This file is deprecated. Use mcp.middleware.auth instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.middleware.auth instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.middleware.auth import *
