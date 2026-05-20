"""
Backward compatibility re-export for mcp_connection_pool.py

This file is deprecated. Use mcp.utils.connection_pool instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.utils.connection_pool instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.utils.connection_pool import *
