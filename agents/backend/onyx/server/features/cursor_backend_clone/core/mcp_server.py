"""
Backward compatibility re-export for mcp_server.py

This file is deprecated. Use mcp.server instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.server instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.server import *
