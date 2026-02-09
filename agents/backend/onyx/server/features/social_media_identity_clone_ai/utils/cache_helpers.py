"""
Helper functions for cache key generation.
Eliminates repetitive hashlib.md5 patterns throughout the codebase.
"""

import hashlib
from typing import Any, Union, List, Dict


def generate_cache_key(*parts: Union[str, int, float, None]) -> str:
    """
    Genera una clave de caché consistente a partir de múltiples partes.
    
    Args:
        *parts: Componentes para la clave (strings, números, o None)
        
    Returns:
        Hash MD5 hexadecimal de la clave generada
        
    Examples:
        >>> generate_cache_key("extract_profile", "tiktok", "username123")
        'a1b2c3d4e5f6...'
        
        >>> generate_cache_key("identity", identity_id, "v2")
        'f6e5d4c3b2a1...'
    """
    # Filtrar None y convertir todo a string
    parts_str = [str(p) if p is not None else "" for p in parts]
    key_string = "_".join(parts_str)
    return hashlib.md5(key_string.encode()).hexdigest()


def generate_cache_key_from_dict(prefix: str, data: Dict[str, Any]) -> str:
    """
    Genera clave de caché desde un diccionario ordenado.
    
    Args:
        prefix: Prefijo para la clave
        data: Diccionario con datos a incluir
        
    Returns:
        Hash MD5 hexadecimal
        
    Examples:
        >>> generate_cache_key_from_dict("profile", {"platform": "tiktok", "username": "user123"})
        'a1b2c3d4e5f6...'
    """
    # Ordenar keys para consistencia
    sorted_items = sorted(data.items())
    parts = [prefix] + [f"{k}:{v}" for k, v in sorted_items]
    return generate_cache_key(*parts)








