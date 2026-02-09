"""
API endpoints para búsqueda y filtrado avanzado
"""

import logging
from typing import Optional
from fastapi import APIRouter, Query, Depends, HTTPException, status
from pydantic import BaseModel

try:
    from api.dependencies import SongServiceDep
    from api.filters import SongFilters, apply_filters
    from api.pagination import PaginationParams, create_paginated_response
except ImportError:
    from .dependencies import SongServiceDep
    from .filters import SongFilters, apply_filters
    from .pagination import PaginationParams, create_paginated_response

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/suno",
    tags=["search"]
)


@router.get("/search/songs")
async def search_songs(
    query: Optional[str] = Query(None, description="Búsqueda por texto"),
    filters: SongFilters = Depends(),
    pagination: PaginationParams = Depends(),
    song_service: SongServiceDep = Depends()
):
    """Búsqueda avanzada de canciones con filtros y paginación"""
    # Obtener todas las canciones (o aplicar filtro básico)
    songs = song_service.list_songs(
        user_id=filters.user_id,
        limit=1000,  # Obtener más para filtrar
        offset=0
    )
    
    # Aplicar filtros
    if filters.to_dict():
        songs = apply_filters(songs, filters)
    
    # Búsqueda por texto si se proporciona
    if query:
        query_lower = query.lower()
        songs = [
            song for song in songs
            if query_lower in song.get("prompt", "").lower()
            or query_lower in song.get("song_id", "").lower()
        ]
    
    # Aplicar paginación
    total = len(songs)
    paginated_songs = songs[pagination.offset:pagination.offset + pagination.limit]
    
    return create_paginated_response(
        items=paginated_songs,
        total=total,
        limit=pagination.limit,
        offset=pagination.offset
    )

