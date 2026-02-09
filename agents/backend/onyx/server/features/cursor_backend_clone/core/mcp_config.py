"""
Backward compatibility re-export for mcp_config.py

This file is deprecated. Use mcp.config instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.config instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.config import *
