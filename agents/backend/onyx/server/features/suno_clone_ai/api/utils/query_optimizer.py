"""
Utilidades para optimizar queries y búsquedas

Incluye funciones para optimizar búsquedas y filtros.
"""

from typing import List, Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def optimize_search_query(query: str) -> str:
    """
    Optimiza una query de búsqueda.
    
    Args:
        query: Query original
        
    Returns:
        Query optimizada
    """
    # Normalizar espacios
    query = " ".join(query.split())
    
    # Limitar longitud
    if len(query) > 200:
        query = query[:200]
    
    return query.strip()


def filter_songs_efficiently(
    songs: List[Dict[str, Any]],
    filters: Dict[str, Any],
    query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filtra canciones de forma eficiente usando generadores.
    
    Args:
        songs: Lista de canciones
        filters: Diccionario de filtros a aplicar
        query: Query de búsqueda opcional
        
    Returns:
        Lista de canciones filtradas
    """
    # Guard clause: si no hay filtros ni query, retornar todo
    if not filters and not query:
        return songs
    
    query_lower = query.lower() if query else None
    filtered = []
    
    for song in songs:
        # Aplicar query de búsqueda
        if query_lower:
            prompt = song.get("prompt", "").lower()
            metadata_str = str(song.get("metadata", {})).lower()
            if query_lower not in prompt and query_lower not in metadata_str:
                continue
        
        # Aplicar filtros
        if not _matches_filters(song, filters):
            continue
        
        filtered.append(song)
    
    return filtered


def _matches_filters(song: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Verifica si una canción coincide con los filtros"""
    # Filtro por género
    if "genre" in filters and filters["genre"]:
        song_genre = song.get("metadata", {}).get("genre", "").lower()
        if filters["genre"].lower() not in song_genre:
            return False
    
    # Filtro por estado
    if "status" in filters and filters["status"]:
        if song.get("status") != filters["status"]:
            return False
    
    # Filtro por rating mínimo
    if "min_rating" in filters and filters["min_rating"] is not None:
        song_rating = song.get("metadata", {}).get("average_rating", 0.0)
        if song_rating < filters["min_rating"]:
            return False
    
    # Filtro por tags
    if "tags" in filters and filters["tags"]:
        song_tags = [t.lower() for t in song.get("metadata", {}).get("tags", [])]
        required_tags = [t.lower() for t in filters["tags"]]
        if not all(tag in song_tags for tag in required_tags):
            return False
    
    # Filtro por user_id
    if "user_id" in filters and filters["user_id"]:
        if song.get("user_id") != filters["user_id"]:
            return False
    
    return True


def paginate_results(
    items: List[Any],
    limit: int,
    offset: int
) -> Dict[str, Any]:
    """
    Pagina resultados de forma eficiente.
    
    Args:
        items: Lista de items
        limit: Límite de resultados
        offset: Offset para paginación
        
    Returns:
        Diccionario con resultados paginados y metadatos
    """
    total = len(items)
    paginated = items[offset:offset + limit]
    has_more = (offset + limit) < total
    
    return {
        "items": paginated,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
        "page": (offset // limit) + 1 if limit > 0 else 1,
        "total_pages": (total + limit - 1) // limit if limit > 0 else 1
    }

