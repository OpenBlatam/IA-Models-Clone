"""
External integrations for Multi-Model API
"""

from .openrouter import (
    OpenRouterClient,
    get_openrouter_client,
    openrouter_handler,
    POPULAR_OPENROUTER_MODELS
)

__all__ = [
    "OpenRouterClient",
    "get_openrouter_client",
    "openrouter_handler",
    "POPULAR_OPENROUTER_MODELS"
]

