"""
Endpoints para gestión básica de canciones (CRUD)

Optimizado para máximo rendimiento con:
- Servicio async
- Response caching
- Lazy loading
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status, Depends, Response
from fastapi.responses import FileResponse

from ..schemas import SongListResponse
from ..dependencies import SongServiceDep
from ..helpers import get_audio_file_path
from ..helpers.service_helpers import get_song_async_or_sync
from ..validators import validate_song_id, ensure_song_exists, ensure_audio_file_exists
from ..business_logic import get_media_type_from_path
from ..utils.response_cache import cache_response
from ..exceptions import SongNotFoundError, AudioFileNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/songs",
    tags=["songs"]
)


@router.get("", response_model=SongListResponse)
@cache_response(ttl=30)  # Cache por 30 segundos
async def list_songs(
    user_id: Optional[str] = Query(None, description="Filtrar por usuario"),
    limit: int = Query(50, ge=1, le=100, description="Número máximo de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    response: Optional[Response] = None,
    song_service: SongServiceDep = Depends()
) -> SongListResponse:
    """
    Lista todas las canciones generadas con paginación (optimizado).
    
    Usa caching y optimizaciones de rendimiento.
    """
    # Usar helper para obtener servicio async o sync
    songs = await get_song_async_or_sync(
        song_service,
        'list_songs',
        user_id=user_id,
        limit=limit,
        offset=offset
    )
    
    # Headers de cache para el cliente
    if response:
        response.headers["Cache-Control"] = "public, max-age=30"
    
    return SongListResponse(songs=songs, total=len(songs))


@router.get("/{song_id}", response_model=Dict[str, Any])
@cache_response(ttl=60)  # Cache por 60 segundos (datos más estables)
async def get_song(
    song_id: str,
    response: Optional[Response] = None,
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene información de una canción específica (optimizado).
    
    Usa caching para mejorar rendimiento en lecturas frecuentes.
    """
    validate_song_id(song_id)
    
    # Usar helper para obtener servicio async o sync
    song = await get_song_async_or_sync(song_service, 'get_song', song_id)
    result = ensure_song_exists(song, song_id)
    
    # Headers de cache
    if response:
        response.headers["Cache-Control"] = "public, max-age=60"
    
    return result


@router.get("/{song_id}/download")
async def download_song(
    song_id: str,
    song_service: SongServiceDep = Depends()
) -> FileResponse:
    """
    Descarga el archivo de audio de una canción (optimizado).
    
    Usa streaming para archivos grandes y headers de cache.
    """
    validate_song_id(song_id)
    
    # Usar helper para obtener servicio async o sync
    song = await get_song_async_or_sync(song_service, 'get_song', song_id)
    song = ensure_song_exists(song, song_id)
    
    # Guard clause: verificar que existe file_path
    file_path_str = song.get("file_path", "")
    if not file_path_str:
        raise AudioFileNotFoundError(f"File path not found for song {song_id}")
    
    ensure_audio_file_exists(file_path_str, song_id)
    
    file_path = Path(file_path_str)
    media_type = get_media_type_from_path(file_path)
    
    logger.debug(f"Downloading song {song_id} from {file_path}")
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=f"{song_id}{file_path.suffix}",
        headers={
            "Content-Disposition": f'attachment; filename="{song_id}{file_path.suffix}"',
            "Cache-Control": "public, max-age=3600",  # Cache por 1 hora
            "Accept-Ranges": "bytes"  # Soporte para range requests
        }
    )


@router.delete("/{song_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_song(
    song_id: str,
    song_service: SongServiceDep = Depends()
) -> None:
    """
    Elimina una canción (optimizado).
    
    Usa servicio async si está disponible para mejor rendimiento.
    """
    validate_song_id(song_id)
    
    # Verificar existencia usando helper
    song = await get_song_async_or_sync(song_service, 'get_song', song_id)
    
    # Guard clause: verificar que existe antes de eliminar
    if not song:
        raise SongNotFoundError(song_id)
    
    # Eliminar usando helper
    success = await get_song_async_or_sync(song_service, 'delete_song', song_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete song {song_id}"
        )
    
    logger.info(f"Song {song_id} deleted successfully")

