"""
Endpoints para sistema de tags y etiquetas

Este módulo proporciona funcionalidades para organizar canciones mediante tags:
- Agregar/eliminar tags a canciones
- Buscar canciones por tags
- Obtener estadísticas de tags
- Gestión de tags populares

Características:
- Validación de tags
- Normalización automática (lowercase)
- Búsqueda eficiente
- Estadísticas de uso
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, status, Depends

from ..dependencies import SongServiceDep
from ..validators import validate_song_id, ensure_song_exists

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/songs",
    tags=["tags"],
    responses={
        400: {"description": "Bad request - Parámetros inválidos"},
        404: {"description": "Not found - Canción no encontrada"},
        500: {"description": "Internal server error"}
    }
)


@router.post(
    "/{song_id}/tags",
    summary="Agregar tags",
    description="Agrega tags/etiquetas a una canción para facilitar la organización"
)
async def add_tags_to_song(
    song_id: str,
    tags: str = Query(..., min_length=1, max_length=500, description="Tags separados por coma"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Agrega tags/etiquetas a una canción para facilitar la organización.
    
    Los tags se normalizan automáticamente (lowercase, sin espacios extra).
    Si un tag ya existe, no se duplica.
    
    Args:
        song_id: ID único de la canción (UUID)
        tags: Tags separados por coma (ej: "rock,energetic,fast")
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Mensaje de confirmación con todos los tags de la canción
    
    Raises:
        HTTPException 400: Si no se proporcionan tags o el formato es inválido
        HTTPException 404: Si la canción no existe
    
    Example:
        ```
        POST /suno/songs/123e4567-e89b-12d3-a456-426614174000/tags?tags=rock,energetic,fast
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        # Parsear y normalizar tags
        tag_list = [
            tag.strip().lower().replace(" ", "-")
            for tag in tags.split(",")
            if tag.strip()
        ]
        
        # Validar tags
        if not tag_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one tag is required"
            )
        
        # Validar longitud de cada tag
        for tag in tag_list:
            if len(tag) > 50:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tag '{tag}' exceeds maximum length of 50 characters"
                )
        
        # Actualizar metadata
        metadata = song.get("metadata", {})
        existing_tags = set(metadata.get("tags", []))
        new_tags = set(tag_list) - existing_tags
        existing_tags.update(tag_list)
        metadata["tags"] = sorted(list(existing_tags))  # Ordenar para consistencia
        
        song_service.save_song(
            song_id=song_id,
            user_id=song.get("user_id"),
            prompt=song.get("prompt", ""),
            file_path=song.get("file_path", ""),
            metadata=metadata
        )
        
        logger.info(f"Tags {list(new_tags)} added to song {song_id} (total: {len(existing_tags)})")
        
        return {
            "message": "Tags added successfully",
            "song_id": song_id,
            "added_tags": list(new_tags),
            "all_tags": list(existing_tags),
            "total_tags": len(existing_tags)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding tags to song: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding tags: {str(e)}"
        )


@router.delete("/{song_id}/tags")
async def remove_tags_from_song(
    song_id: str,
    tags: str = Query(..., description="Tags a eliminar separados por coma"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """Elimina tags de una canción"""
    validate_song_id(song_id)
    song = ensure_song_exists(song_service.get_song(song_id), song_id)
    
    # Parsear tags a eliminar
    tags_to_remove = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
    
    metadata = song.get("metadata", {})
    existing_tags = set(metadata.get("tags", []))
    existing_tags.difference_update(tags_to_remove)
    metadata["tags"] = list(existing_tags)
    
    song_service.save_song(
        song_id=song_id,
        user_id=song.get("user_id"),
        prompt=song.get("prompt", ""),
        file_path=song.get("file_path", ""),
        metadata=metadata
    )
    
    return {
        "message": "Tags removed successfully",
        "song_id": song_id,
        "tags": list(existing_tags)
    }


@router.get(
    "/by-tag/{tag}",
    summary="Buscar canciones por tag",
    description="Obtiene canciones que tienen un tag específico"
)
async def get_songs_by_tag(
    tag: str,
    limit: int = Query(50, ge=1, le=100, description="Número máximo de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    sort_by: Optional[str] = Query("created_at", description="Ordenar por: created_at, rating, popularity"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene canciones filtradas por tag.
    
    Args:
        tag: Tag a buscar (case-insensitive)
        limit: Número máximo de resultados
        offset: Offset para paginación
        sort_by: Campo para ordenar (created_at, rating, popularity)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Lista de canciones con el tag especificado
    
    Example:
        ```
        GET /suno/songs/by-tag/rock?limit=20&sort_by=rating
        ```
    """
    try:
        all_songs = song_service.list_songs(limit=10000, offset=0)
        tag_lower = tag.lower().strip()
        
        filtered_songs = [
            song for song in all_songs
            if tag_lower in [t.lower() for t in song.get("metadata", {}).get("tags", [])]
        ]
        
        # Ordenar si se especifica
        if sort_by == "rating":
            filtered_songs.sort(
                key=lambda x: x.get("metadata", {}).get("average_rating", 0.0),
                reverse=True
            )
        elif sort_by == "popularity":
            filtered_songs.sort(
                key=lambda x: (
                    len(x.get("metadata", {}).get("favorites", [])) * 2 +
                    x.get("metadata", {}).get("average_rating", 0.0) * 3
                ),
                reverse=True
            )
        # created_at es el default, ya viene ordenado
        
        total = len(filtered_songs)
        paginated_songs = filtered_songs[offset:offset + limit]
        has_more = (offset + limit) < total
        
        return {
            "tag": tag,
            "songs": paginated_songs,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "sort_by": sort_by
        }
    except Exception as e:
        logger.error(f"Error getting songs by tag {tag}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving songs by tag: {str(e)}"
        )


@router.get("/tags/popular")
async def get_popular_tags(
    limit: int = Query(20, ge=1, le=100, description="Número de tags a retornar"),
    min_usage: int = Query(1, ge=1, description="Uso mínimo del tag"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene los tags más populares del sistema.
    
    Args:
        limit: Número de tags a retornar
        min_usage: Uso mínimo requerido para aparecer
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Lista de tags ordenados por popularidad
    
    Example:
        ```
        GET /suno/songs/tags/popular?limit=30&min_usage=5
        ```
    """
    try:
        all_songs = song_service.list_songs(limit=10000, offset=0)
        
        tag_usage = {}
        for song in all_songs:
            tags = song.get("metadata", {}).get("tags", [])
            for tag in tags:
                tag_lower = tag.lower()
                tag_usage[tag_lower] = tag_usage.get(tag_lower, 0) + 1
        
        # Filtrar por uso mínimo y ordenar
        popular_tags = [
            {"tag": tag, "usage_count": count}
            for tag, count in tag_usage.items()
            if count >= min_usage
        ]
        popular_tags.sort(key=lambda x: x["usage_count"], reverse=True)
        
        return {
            "popular_tags": popular_tags[:limit],
            "total_tags": len(tag_usage),
            "tags_meeting_criteria": len(popular_tags),
            "min_usage": min_usage
        }
    except Exception as e:
        logger.error(f"Error getting popular tags: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting popular tags: {str(e)}"
        )


@router.get("/{song_id}/tags")
async def get_song_tags(
    song_id: str,
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene todos los tags de una canción.
    
    Args:
        song_id: ID único de la canción (UUID)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Lista de tags de la canción
    
    Example:
        ```
        GET /suno/songs/123e4567-e89b-12d3-a456-426614174000/tags
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        tags = song.get("metadata", {}).get("tags", [])
        
        return {
            "song_id": song_id,
            "tags": sorted(tags),
            "tag_count": len(tags)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting song tags: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting song tags: {str(e)}"
        )

