"""
Backward compatibility re-export for mcp_request_queue.py

This file is deprecated. Use mcp.utils.request_queue instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.utils.request_queue instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.utils.request_queue import *
