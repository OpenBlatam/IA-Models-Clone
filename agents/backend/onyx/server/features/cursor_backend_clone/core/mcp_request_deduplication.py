"""
Backward compatibility re-export for mcp_request_deduplication.py

This file is deprecated. Use mcp.middleware.request_deduplication instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.middleware.request_deduplication instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.middleware.request_deduplication import *
