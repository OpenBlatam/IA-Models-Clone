"""
Validadores para el sistema de análisis musical
"""

import re
from typing import Optional
from .exceptions import InvalidTrackIDException


def validate_spotify_track_id(track_id: str) -> bool:
    """
    Valida que un track_id de Spotify tenga el formato correcto
    
    Los track IDs de Spotify son strings alfanuméricos de 22 caracteres
    """
    if not track_id:
        raise InvalidTrackIDException("Track ID no puede estar vacío")
    
    if not isinstance(track_id, str):
        raise InvalidTrackIDException("Track ID debe ser un string")
    
    # Spotify track IDs son alfanuméricos y tienen 22 caracteres
    pattern = r'^[a-zA-Z0-9]{22}$'
    
    if not re.match(pattern, track_id):
        raise InvalidTrackIDException(
            f"Track ID inválido: '{track_id}'. Debe ser alfanumérico de 22 caracteres"
        )
    
    return True


def validate_search_query(query: str) -> bool:
    """Valida una query de búsqueda"""
    if not query:
        raise ValueError("Query de búsqueda no puede estar vacía")
    
    if len(query.strip()) < 2:
        raise ValueError("Query de búsqueda debe tener al menos 2 caracteres")
    
    if len(query) > 200:
        raise ValueError("Query de búsqueda no puede exceder 200 caracteres")
    
    return True


def sanitize_search_query(query: str) -> str:
    """Sanitiza una query de búsqueda"""
    # Eliminar caracteres especiales peligrosos
    query = query.strip()
    # Permitir letras, números, espacios y algunos caracteres especiales comunes
    query = re.sub(r'[^\w\s\-\.\']', '', query)
    return query[:200]  # Limitar longitud

