"""
Backward compatibility re-export for mcp_errors.py

This file is deprecated. Use mcp.errors instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.errors instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.errors import *
