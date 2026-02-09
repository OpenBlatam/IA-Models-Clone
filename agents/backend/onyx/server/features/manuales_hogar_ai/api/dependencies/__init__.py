"""
API Dependencies Module
=======================

Dependencies para inyección en endpoints.
"""

from .manual_dependencies import (
    get_manual_generator,
    get_openrouter_client,
)

__all__ = [
    "get_manual_generator",
    "get_openrouter_client",
]

