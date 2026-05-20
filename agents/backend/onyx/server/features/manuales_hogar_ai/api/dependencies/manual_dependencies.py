"""
Manual Dependencies
===================

Dependencies para endpoints de manuales.
"""

from ...core.manual_generator import ManualGenerator
from ...infrastructure.openrouter.openrouter_client import OpenRouterClient


def get_manual_generator() -> ManualGenerator:
    """Obtener instancia de ManualGenerator."""
    client = OpenRouterClient()
    return ManualGenerator(openrouter_client=client)


def get_openrouter_client() -> OpenRouterClient:
    """Obtener instancia de OpenRouterClient."""
    return OpenRouterClient()

