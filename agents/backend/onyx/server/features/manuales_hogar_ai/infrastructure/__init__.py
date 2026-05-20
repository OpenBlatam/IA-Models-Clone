"""Infraestructura del módulo."""

from .openrouter.openrouter_client import OpenRouterClient
from .openrouter_client import OpenRouterClient as OpenRouterClientLegacy

__all__ = [
    "OpenRouterClient",
    "OpenRouterClientLegacy",
]




