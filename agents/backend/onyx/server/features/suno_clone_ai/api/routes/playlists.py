"""
Endpoints para gestión de playlists

Este módulo proporciona funcionalidades completas para crear y gestionar playlists:
- Crear playlists personalizadas
- Agregar/eliminar canciones
- Obtener información de playlists
- Listar playlists de usuario
- Compartir playlists

Características:
- Validación robusta
- Manejo de duplicados
- Ordenamiento de canciones
- Estadísticas de playlists
"""

import logging
import uuid
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query, status, Depends

from ..dependencies import SongServiceDep
from ..validators import validate_song_id, ensure_song_exists

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/playlists",
    tags=["playlists"],
    responses={
        400: {"description": "Bad request - Parámetros inválidos"},
        404: {"description": "Not found - Playlist o canción no encontrada"},
        500: {"description": "Internal server error"}
    }
)


@router.post(
    "",
    summary="Crear playlist",
    description="Crea una nueva playlist vacía para un usuario"
)
async def create_playlist(
    name: str = Query(..., min_length=1, max_length=100, description="Nombre de la playlist"),
    description: Optional[str] = Query(None, max_length=500, description="Descripción de la playlist"),
    user_id: str = Query(..., min_length=1, max_length=100, description="ID del usuario"),
    is_public: bool = Query(False, description="Si la playlist es pública"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Crea una nueva playlist vacía.
    
    Args:
        name: Nombre de la playlist (1-100 caracteres)
        description: Descripción opcional (máx 500 caracteres)
        user_id: ID del usuario propietario
        is_public: Si la playlist es pública (otros usuarios pueden verla)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Información de la playlist creada
    
    Raises:
        HTTPException 400: Si los parámetros son inválidos
    
    Example:
        ```
        POST /suno/playlists?name=My Favorites&description=Best songs&user_id=user123&is_public=true
        ```
    """
    try:
        playlist_id = str(uuid.uuid4())
        
        playlist_data = {
            "playlist_id": playlist_id,
            "name": name,
            "description": description,
            "user_id": user_id,
            "is_public": is_public,
            "songs": [],
            "song_count": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "play_count": 0,
            "favorites_count": 0
        }
        
        # En producción, esto se guardaría en una tabla de playlists
        song_service.save_song(
            song_id=f"playlist_{playlist_id}",
            user_id=user_id,
            prompt=f"Playlist: {name}",
            file_path="",
            metadata=playlist_data
        )
        
        logger.info(f"Playlist {playlist_id} created by user {user_id} (public: {is_public})")
        
        return {
            "message": "Playlist created successfully",
            "playlist_id": playlist_id,
            "playlist": playlist_data
        }
    except Exception as e:
        logger.error(f"Error creating playlist: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating playlist: {str(e)}"
        )


@router.post(
    "/{playlist_id}/songs/{song_id}",
    summary="Agregar canción a playlist",
    description="Agrega una canción a una playlist existente"
)
async def add_song_to_playlist(
    playlist_id: str,
    song_id: str,
    position: Optional[int] = Query(None, ge=0, description="Posición en la playlist (opcional)"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Agrega una canción a una playlist.
    
    Si la canción ya está en la playlist, la operación es idempotente.
    Se puede especificar una posición para insertar en un lugar específico.
    
    Args:
        playlist_id: ID de la playlist
        song_id: ID de la canción a agregar (UUID)
        position: Posición opcional en la playlist (0 = inicio)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Mensaje de confirmación con información actualizada
    
    Raises:
        HTTPException 400: Si el song_id es inválido
        HTTPException 404: Si la playlist o canción no existe
    
    Example:
        ```
        POST /suno/playlists/playlist123/songs/song456?position=0
        ```
    """
    try:
        validate_song_id(song_id)
        ensure_song_exists(song_service.get_song(song_id), song_id)
        
        # Obtener playlist
        playlist = song_service.get_song(f"playlist_{playlist_id}")
        if not playlist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Playlist {playlist_id} not found"
            )
        
        metadata = playlist.get("metadata", {})
        songs = metadata.get("songs", [])
        
        was_already_in_playlist = song_id in songs
        
        if not was_already_in_playlist:
            if position is not None and 0 <= position <= len(songs):
                songs.insert(position, song_id)
            else:
                songs.append(song_id)
            
            metadata["songs"] = songs
            metadata["song_count"] = len(songs)
            metadata["updated_at"] = datetime.now().isoformat()
            
            song_service.save_song(
                song_id=f"playlist_{playlist_id}",
                user_id=playlist.get("user_id"),
                prompt=playlist.get("prompt", ""),
                file_path="",
                metadata=metadata
            )
            
            logger.info(f"Song {song_id} added to playlist {playlist_id} at position {position or len(songs)-1}")
        else:
            logger.debug(f"Song {song_id} already in playlist {playlist_id}")
        
        return {
            "message": "Song added to playlist" if not was_already_in_playlist else "Song already in playlist",
            "playlist_id": playlist_id,
            "song_id": song_id,
            "position": position if position is not None else len(songs) - 1,
            "song_count": len(songs),
            "was_already_in_playlist": was_already_in_playlist
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding song to playlist: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding song to playlist: {str(e)}"
        )


@router.delete("/{playlist_id}/songs/{song_id}")
async def remove_song_from_playlist(
    playlist_id: str,
    song_id: str,
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Elimina una canción de una playlist.
    
    Args:
        playlist_id: ID de la playlist
        song_id: ID de la canción a eliminar
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Mensaje de confirmación
    
    Example:
        ```
        DELETE /suno/playlists/playlist123/songs/song456
        ```
    """
    try:
        validate_song_id(song_id)
        
        playlist = song_service.get_song(f"playlist_{playlist_id}")
        if not playlist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Playlist {playlist_id} not found"
            )
        
        metadata = playlist.get("metadata", {})
        songs = metadata.get("songs", [])
        
        if song_id in songs:
            songs.remove(song_id)
            metadata["songs"] = songs
            metadata["song_count"] = len(songs)
            metadata["updated_at"] = datetime.now().isoformat()
            
            song_service.save_song(
                song_id=f"playlist_{playlist_id}",
                user_id=playlist.get("user_id"),
                prompt=playlist.get("prompt", ""),
                file_path="",
                metadata=metadata
            )
            
            logger.info(f"Song {song_id} removed from playlist {playlist_id}")
        
        return {
            "message": "Song removed from playlist",
            "playlist_id": playlist_id,
            "song_id": song_id,
            "song_count": len(songs)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing song from playlist: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing song from playlist: {str(e)}"
        )


@router.get("/users/{user_id}")
async def get_user_playlists(
    user_id: str,
    limit: int = Query(50, ge=1, le=100, description="Número máximo de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene todas las playlists de un usuario.
    
    Args:
        user_id: ID del usuario
        limit: Número máximo de resultados
        offset: Offset para paginación
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Lista de playlists del usuario
    
    Example:
        ```
        GET /suno/playlists/users/user123?limit=20
        ```
    """
    try:
        all_songs = song_service.list_songs(limit=10000, offset=0)
        
        user_playlists = []
        for song in all_songs:
            if song.get("song_id", "").startswith("playlist_"):
                metadata = song.get("metadata", {})
                if metadata.get("user_id") == user_id:
                    user_playlists.append({
                        "playlist_id": metadata.get("playlist_id"),
                        "name": metadata.get("name"),
                        "description": metadata.get("description"),
                        "song_count": metadata.get("song_count", 0),
                        "created_at": metadata.get("created_at"),
                        "updated_at": metadata.get("updated_at"),
                        "is_public": metadata.get("is_public", False)
                    })
        
        total = len(user_playlists)
        paginated_playlists = user_playlists[offset:offset + limit]
        has_more = (offset + limit) < total
        
        return {
            "user_id": user_id,
            "playlists": paginated_playlists,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more
        }
    except Exception as e:
        logger.error(f"Error getting user playlists: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting user playlists: {str(e)}"
        )


@router.get("/{playlist_id}")
async def get_playlist(
    playlist_id: str,
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """Obtiene información de una playlist con sus canciones"""
    playlist = song_service.get_song(f"playlist_{playlist_id}")
    if not playlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Playlist {playlist_id} not found"
        )
    
    metadata = playlist.get("metadata", {})
    song_ids = metadata.get("songs", [])
    
    # Obtener información de las canciones
    songs = []
    for song_id in song_ids:
        song = song_service.get_song(song_id)
        if song:
            songs.append(song)
    
    return {
        "playlist": metadata,
        "songs": songs,
        "song_count": len(songs)
    }

