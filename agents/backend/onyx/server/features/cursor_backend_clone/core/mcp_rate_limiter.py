"""
Backward compatibility re-export for mcp_rate_limiter.py

This file is deprecated. Use mcp.middleware.rate_limiter instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.middleware.rate_limiter instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.middleware.rate_limiter import *
