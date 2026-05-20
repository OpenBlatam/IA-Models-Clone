"""
Backward compatibility re-export for mcp_events.py

This file is deprecated. Use mcp.events instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.events instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.events import *
