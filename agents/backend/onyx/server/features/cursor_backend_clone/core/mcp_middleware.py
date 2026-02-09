"""
Backward compatibility re-export for mcp_middleware.py

This file is deprecated. Use mcp.middleware.middleware instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.middleware.middleware instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.middleware.middleware import *
