"""
Backward compatibility re-export for mcp_prometheus.py

This file is deprecated. Use mcp.metrics.prometheus instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.metrics.prometheus instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.metrics.prometheus import *
