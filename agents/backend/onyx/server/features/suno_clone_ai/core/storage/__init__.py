"""
Sistema de Almacenamiento

Proporciona abstracciones para diferentes backends de almacenamiento.
"""

from .local_storage import LocalStorage

__all__ = [
    "LocalStorage"
]

