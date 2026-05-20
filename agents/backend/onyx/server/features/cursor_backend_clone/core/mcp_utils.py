"""
Backward compatibility re-export for mcp_utils.py

This file is deprecated. Use mcp.utils.utils instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.utils.utils instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.utils.utils import *
