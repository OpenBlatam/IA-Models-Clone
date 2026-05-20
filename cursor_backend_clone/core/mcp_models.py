"""
Backward compatibility re-export for mcp_models.py

This file is deprecated. Use mcp.models instead.
"""
import warnings

warnings.warn(
    "{name} is deprecated. Use mcp.models instead.",
    DeprecationWarning,
    stacklevel=2
)

from .mcp.models import *
