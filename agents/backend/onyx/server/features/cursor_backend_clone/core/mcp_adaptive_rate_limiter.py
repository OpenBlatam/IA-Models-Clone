"""
Backward compatibility re-export for mcp_adaptive_rate_limiter.py

This file is deprecated. Use mcp.middleware.adaptive_rate_limiter instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.middleware.adaptive_rate_limiter instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.middleware.adaptive_rate_limiter import *
