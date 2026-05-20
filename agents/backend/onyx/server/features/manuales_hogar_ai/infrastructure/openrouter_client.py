"""
OpenRouter Client (Legacy)
==========================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar infrastructure.openrouter.openrouter_client.OpenRouterClient
"""

from .openrouter.openrouter_client import OpenRouterClient

__all__ = ["OpenRouterClient"]
