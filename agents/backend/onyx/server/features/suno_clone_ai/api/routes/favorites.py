"""
Endpoints para favoritos y sistema de ratings

Este módulo proporciona funcionalidades para:
- Agregar/eliminar canciones de favoritos
- Calificar canciones (0-5 estrellas)
- Obtener estadísticas de favoritos y ratings
- Gestión de preferencias de usuario

Características:
- Validación robusta de inputs
- Cálculo automático de promedios
- Manejo de errores completo
- Logging detallado
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, status

from ..dependencies import SongServiceDep
from ..validators import validate_song_id, ensure_song_exists

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/songs",
    tags=["favorites"],
    responses={
        400: {"description": "Bad request - Parámetros inválidos"},
        404: {"description": "Not found - Canción no encontrada"},
        500: {"description": "Internal server error"}
    }
)


@router.post(
    "/{song_id}/favorite",
    summary="Agregar a favoritos",
    description="Agrega una canción a los favoritos de un usuario"
)
async def add_to_favorites(
    song_id: str,
    user_id: str = Query(..., min_length=1, max_length=100, description="ID del usuario"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Agrega una canción a los favoritos de un usuario.
    
    Si la canción ya está en favoritos, la operación es idempotente
    y no produce error.
    
    Args:
        song_id: ID único de la canción (UUID)
        user_id: ID del usuario
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Mensaje de confirmación con información actualizada
    
    Raises:
        HTTPException 400: Si el formato del song_id es inválido
        HTTPException 404: Si la canción no existe
    
    Example:
        ```
        POST /suno/songs/123e4567-e89b-12d3-a456-426614174000/favorite?user_id=user123
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        # Actualizar metadata con favorito
        metadata = song.get("metadata", {})
        favorites = metadata.get("favorites", [])
        
        was_already_favorite = user_id in favorites
        
        if not was_already_favorite:
            favorites.append(user_id)
            metadata["favorites"] = favorites
            metadata["favorites_count"] = len(favorites)
            
            song_service.save_song(
                song_id=song_id,
                user_id=song.get("user_id"),
                prompt=song.get("prompt", ""),
                file_path=song.get("file_path", ""),
                metadata=metadata
            )
            
            logger.info(f"Song {song_id} added to favorites for user {user_id}")
        else:
            logger.debug(f"Song {song_id} already in favorites for user {user_id}")
        
        return {
            "message": "Song added to favorites" if not was_already_favorite else "Song already in favorites",
            "song_id": song_id,
            "user_id": user_id,
            "favorites_count": len(favorites),
            "was_already_favorite": was_already_favorite
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding song to favorites: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding to favorites: {str(e)}"
        )


@router.delete("/{song_id}/favorite")
async def remove_from_favorites(
    song_id: str,
    user_id: str = Query(..., description="ID del usuario"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """Elimina una canción de los favoritos de un usuario"""
    validate_song_id(song_id)
    song = ensure_song_exists(song_service.get_song(song_id), song_id)
    
    metadata = song.get("metadata", {})
    favorites = metadata.get("favorites", [])
    if user_id in favorites:
        favorites.remove(user_id)
        metadata["favorites"] = favorites
        song_service.save_song(
            song_id=song_id,
            user_id=song.get("user_id"),
            prompt=song.get("prompt", ""),
            file_path=song.get("file_path", ""),
            metadata=metadata
        )
    
    return {"message": "Song removed from favorites", "song_id": song_id, "user_id": user_id}


@router.get("/{song_id}/favorites")
async def get_favorites_count(
    song_id: str,
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """Obtiene el número de usuarios que han marcado una canción como favorita"""
    validate_song_id(song_id)
    song = ensure_song_exists(song_service.get_song(song_id), song_id)
    
    favorites = song.get("metadata", {}).get("favorites", [])
    return {
        "song_id": song_id,
        "favorites_count": len(favorites),
        "favorited_by": favorites
    }


@router.post(
    "/{song_id}/rate",
    summary="Calificar canción",
    description="Califica una canción con un valor de 0.0 a 5.0 estrellas"
)
async def rate_song(
    song_id: str,
    rating: float = Query(..., ge=0.0, le=5.0, description="Rating de 0.0 a 5.0"),
    user_id: str = Query(..., min_length=1, max_length=100, description="ID del usuario"),
    comment: Optional[str] = Query(None, max_length=500, description="Comentario opcional con el rating"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Califica una canción (0.0 a 5.0 estrellas).
    
    Si el usuario ya calificó la canción, se actualiza su rating anterior.
    El sistema calcula automáticamente el promedio de todos los ratings.
    
    Args:
        song_id: ID único de la canción (UUID)
        rating: Calificación de 0.0 a 5.0
        user_id: ID del usuario que califica
        comment: Comentario opcional con el rating
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Mensaje de confirmación con estadísticas de rating
    
    Raises:
        HTTPException 400: Si el formato es inválido o rating fuera de rango
        HTTPException 404: Si la canción no existe
    
    Example:
        ```
        POST /suno/songs/123e4567-e89b-12d3-a456-426614174000/rate?rating=4.5&user_id=user123&comment=Great song!
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        # Validar rating
        if rating < 0.0 or rating > 5.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Rating must be between 0.0 and 5.0, got {rating}"
            )
        
        metadata = song.get("metadata", {})
        ratings = metadata.get("ratings", {})
        
        # Verificar si el usuario ya calificó
        previous_rating = ratings.get(user_id)
        was_update = previous_rating is not None
        
        # Actualizar rating
        ratings[user_id] = rating
        
        # Guardar comentario si se proporciona
        if comment:
            rating_comments = metadata.get("rating_comments", {})
            rating_comments[user_id] = comment
            metadata["rating_comments"] = rating_comments
        
        # Calcular promedio y estadísticas
        rating_values = list(ratings.values())
        average_rating = sum(rating_values) / len(rating_values) if rating_values else 0.0
        
        # Calcular distribución de ratings
        rating_distribution = {i: 0 for i in range(6)}
        for r in rating_values:
            rating_distribution[int(r)] = rating_distribution.get(int(r), 0) + 1
        
        metadata["ratings"] = ratings
        metadata["average_rating"] = round(average_rating, 2)
        metadata["total_ratings"] = len(ratings)
        metadata["rating_distribution"] = rating_distribution
        
        song_service.save_song(
            song_id=song_id,
            user_id=song.get("user_id"),
            prompt=song.get("prompt", ""),
            file_path=song.get("file_path", ""),
            metadata=metadata
        )
        
        logger.info(
            f"Song {song_id} rated {rating} by user {user_id} "
            f"(was_update={was_update}, new_avg={average_rating:.2f})"
        )
        
        return {
            "message": "Rating updated" if was_update else "Rating saved",
            "song_id": song_id,
            "user_id": user_id,
            "user_rating": rating,
            "previous_rating": previous_rating,
            "average_rating": round(average_rating, 2),
            "total_ratings": len(ratings),
            "rating_distribution": rating_distribution
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rating song: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error rating song: {str(e)}"
        )


@router.get("/{song_id}/ratings")
async def get_song_ratings(
    song_id: str,
    include_distribution: bool = Query(False, description="Incluir distribución de ratings"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene información detallada de los ratings de una canción.
    
    Args:
        song_id: ID único de la canción (UUID)
        include_distribution: Incluir distribución de ratings (0-5)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Información completa de ratings
    
    Example:
        ```
        GET /suno/songs/123e4567-e89b-12d3-a456-426614174000/ratings?include_distribution=true
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        metadata = song.get("metadata", {})
        ratings = metadata.get("ratings", {})
        rating_comments = metadata.get("rating_comments", {})
        
        rating_values = list(ratings.values())
        average_rating = sum(rating_values) / len(rating_values) if rating_values else 0.0
        
        result = {
            "song_id": song_id,
            "average_rating": round(average_rating, 2),
            "total_ratings": len(ratings),
            "ratings": ratings
        }
        
        if include_distribution:
            rating_distribution = {i: 0 for i in range(6)}
            for r in rating_values:
                rating_distribution[int(r)] = rating_distribution.get(int(r), 0) + 1
            result["rating_distribution"] = rating_distribution
        
        if rating_comments:
            result["rating_comments"] = rating_comments
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting song ratings: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting ratings: {str(e)}"
        )


@router.get("/users/{user_id}/favorites")
async def get_user_favorites(
    user_id: str,
    limit: int = Query(50, ge=1, le=100, description="Número máximo de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene todas las canciones favoritas de un usuario.
    
    Args:
        user_id: ID del usuario
        limit: Número máximo de resultados
        offset: Offset para paginación
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Lista de canciones favoritas del usuario
    
    Example:
        ```
        GET /suno/songs/users/user123/favorites?limit=20
        ```
    """
    try:
        all_songs = song_service.list_songs(limit=10000, offset=0)
        
        favorite_songs = [
            song for song in all_songs
            if user_id in song.get("metadata", {}).get("favorites", [])
        ]
        
        total = len(favorite_songs)
        paginated_songs = favorite_songs[offset:offset + limit]
        has_more = (offset + limit) < total
        
        return {
            "user_id": user_id,
            "favorites": paginated_songs,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more
        }
    except Exception as e:
        logger.error(f"Error getting user favorites: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting user favorites: {str(e)}"
        )

