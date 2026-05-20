"""
Backward compatibility re-export for mcp_metrics.py

This file is deprecated. Use mcp.metrics.metrics instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.metrics.metrics instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.metrics.metrics import *
