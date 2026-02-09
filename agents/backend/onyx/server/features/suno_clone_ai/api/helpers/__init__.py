"""
Helpers para la API

Incluye funciones utilitarias y helpers para operaciones comunes.
"""

from .service_helpers import get_song_async_or_sync, execute_async_operation

__all__ = [
    "get_song_async_or_sync",
    "execute_async_operation"
]

