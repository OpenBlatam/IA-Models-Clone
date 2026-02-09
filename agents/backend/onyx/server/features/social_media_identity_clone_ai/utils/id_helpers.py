"""
Helper functions for ID generation.
Eliminates repetitive uuid.uuid4() patterns.
"""

import uuid
from typing import Optional


def generate_id(prefix: Optional[str] = None) -> str:
    """
    Genera un ID único.
    
    Args:
        prefix: Prefijo opcional para el ID (ej: "identity", "content")
        
    Returns:
        String con ID único
        
    Usage:
        >>> generate_id()
        '550e8400-e29b-41d4-a716-446655440000'
        
        >>> generate_id("identity")
        'identity_550e8400-e29b-41d4-a716-446655440000'
    """
    id_str = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{id_str}"
    return id_str


def generate_short_id(length: int = 8) -> str:
    """
    Genera un ID corto (útil para URLs o códigos).
    
    Args:
        length: Longitud del ID (default: 8)
        
    Returns:
        String con ID corto
        
    Usage:
        >>> generate_short_id()
        'a1b2c3d4'
        
        >>> generate_short_id(12)
        'a1b2c3d4e5f6'
    """
    return uuid.uuid4().hex[:length]








