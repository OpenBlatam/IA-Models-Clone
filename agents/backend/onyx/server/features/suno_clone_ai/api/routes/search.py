"""
Endpoints para búsqueda y filtrado avanzado

Optimizado con:
- Query optimization
- Response caching
- Async operations
- Efficient filtering
"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status, Depends, Response

from ..dependencies import SongServiceDep
from ..helpers.service_helpers import get_song_async_or_sync
from ..utils.response_cache import cache_response
from ..utils.query_optimizer import (
    optimize_search_query,
    filter_songs_efficiently,
    paginate_results
)
from ..exceptions import InvalidInputError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/songs",
    tags=["search"]
)


@router.get("/search")
@cache_response(ttl=60)  # Cache por 60 segundos
async def search_songs(
    query: str = Query(..., min_length=1, max_length=200, description="Término de búsqueda"),
    genre: Optional[str] = Query(None, description="Filtrar por género"),
    status_filter: Optional[str] = Query(None, description="Filtrar por estado"),
    user_id: Optional[str] = Query(None, description="Filtrar por usuario"),
    min_rating: Optional[float] = Query(None, ge=0.0, le=5.0, description="Rating mínimo"),
    tags: Optional[str] = Query(None, description="Tags separados por coma"),
    limit: int = Query(20, ge=1, le=100, description="Número máximo de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    response: Optional[Response] = None,
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Búsqueda avanzada de canciones con múltiples filtros (optimizado).
    
    Busca en prompts, metadatos y otros campos con filtros combinados.
    Usa optimizaciones de query y caching para mejor rendimiento.
    """
    # Optimizar query
    query = optimize_search_query(query)
    
    # Validar status_filter si se proporciona
    if status_filter and status_filter not in ["processing", "completed", "failed"]:
        raise InvalidInputError(
            f"Invalid status filter: {status_filter}. Must be one of: processing, completed, failed"
        )
    
    # Obtener canciones usando servicio async si está disponible
    all_songs = await get_song_async_or_sync(
        song_service,
        'list_songs',
        user_id=user_id,
        limit=10000,
        offset=0
    )
    
    # Preparar filtros
    filters = {
        "genre": genre,
        "status": status_filter,
        "user_id": user_id,
        "min_rating": min_rating,
        "tags": [tag.strip().lower() for tag in tags.split(",")] if tags else None
    }
    
    # Filtrar canciones de forma eficiente
    filtered_songs = filter_songs_efficiently(all_songs, filters, query)
    
    # Aplicar paginación optimizada
    paginated = paginate_results(filtered_songs, limit, offset)
    
    # Headers de cache
    if response:
        response.headers["Cache-Control"] = "public, max-age=60"
    
    return {
        "query": query,
        "songs": paginated["items"],
        "total": paginated["total"],
        "limit": paginated["limit"],
        "offset": paginated["offset"],
        "has_more": paginated["has_more"],
        "page": paginated["page"],
        "total_pages": paginated["total_pages"],
        "filters_applied": {k: v for k, v in filters.items() if v is not None}
    }


@router.get("/by-genre/{genre}")
@cache_response(ttl=120)  # Cache por 2 minutos (géneros cambian menos)
async def get_songs_by_genre(
    genre: str,
    limit: int = Query(50, ge=1, le=100, description="Número máximo de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    response: Optional[Response] = None,
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene canciones filtradas por género musical (optimizado).
    
    Usa caching y optimizaciones de query.
    """
    # Normalizar género
    genre = genre.strip().lower()
    
    # Obtener canciones usando servicio async
    all_songs = await get_song_async_or_sync(
        song_service,
        'list_songs',
        limit=10000,
        offset=0
    )
    
    # Filtrar por género de forma eficiente
    filters = {"genre": genre}
    filtered_songs = filter_songs_efficiently(all_songs, filters)
    
    # Aplicar paginación
    paginated = paginate_results(filtered_songs, limit, offset)
    
    # Headers de cache
    if response:
        response.headers["Cache-Control"] = "public, max-age=120"
    
    return {
        "genre": genre,
        "songs": paginated["items"],
        "total": paginated["total"],
        "limit": paginated["limit"],
        "offset": paginated["offset"],
        "has_more": paginated["has_more"],
        "page": paginated["page"],
        "total_pages": paginated["total_pages"]
    }

