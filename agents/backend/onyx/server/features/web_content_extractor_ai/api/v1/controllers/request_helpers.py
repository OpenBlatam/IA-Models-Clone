"""
Helpers para procesar requests
"""

from typing import List
from pydantic import HttpUrl


def convert_urls_to_strings(urls: List[HttpUrl]) -> List[str]:
    """
    Convierte una lista de HttpUrl a lista de strings.
    
    Args:
        urls: Lista de HttpUrl objects
        
    Returns:
        Lista de strings con las URLs
    """
    return [str(url) for url in urls]


def get_first_url_or_default(urls: List[HttpUrl], default: str = "batch") -> str:
    """
    Obtiene la primera URL de una lista o retorna un valor por defecto.
    
    Args:
        urls: Lista de URLs
        default: Valor por defecto si la lista está vacía
        
    Returns:
        Primera URL como string o el valor por defecto
    """
    return str(urls[0]) if urls else default

