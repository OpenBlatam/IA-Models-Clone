"""
Backward compatibility re-export for mcp_token_bucket.py

This file is deprecated. Use mcp.utils.token_bucket instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.utils.token_bucket instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.utils.token_bucket import *
