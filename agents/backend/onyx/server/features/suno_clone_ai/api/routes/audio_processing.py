"""
Endpoints para procesamiento y edición de audio

Optimizado con:
- Async operations
- Better error handling
- Input validation
"""

import logging
import uuid
from pathlib import Path
from typing import List, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Response

from ..schemas import SongResponse, AudioEditRequest, AudioMixRequest, SongAnalysisResponse
from ..dependencies import SongServiceDep, AudioProcessorDep
from ..helpers import (
    generate_song_id,
    get_audio_file_path,
    load_audio_file,
    save_audio_file,
    apply_audio_operations
)
from ..helpers.service_helpers import get_song_async_or_sync
from ..validators import validate_song_id, ensure_song_exists
from ..utils.response_cache import cache_response
from ..exceptions import SongNotFoundError, AudioProcessingError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/songs",
    tags=["audio-processing"]
)


@router.post("/{song_id}/edit", response_model=SongResponse)
async def edit_song(
    song_id: str,
    edit_request: AudioEditRequest,
    song_service: SongServiceDep = Depends(),
    audio_processor: AudioProcessorDep = Depends()
) -> SongResponse:
    """
    Edita una canción con efectos y procesamiento (optimizado).
    
    Aplica operaciones de audio de forma eficiente y maneja errores robustamente.
    """
    validate_song_id(song_id)
    
    # Obtener canción usando servicio async
    song = await get_song_async_or_sync(song_service, 'get_song', song_id)
    song = ensure_song_exists(song, song_id)
    
    # Guard clause: verificar file_path
    file_path_str = song.get("file_path", "")
    if not file_path_str:
        raise AudioProcessingError(f"File path not found for song {song_id}")
    
    try:
        audio, sample_rate = load_audio_file(file_path_str)
    except Exception as e:
        logger.error(f"Error loading audio file: {e}", exc_info=True)
        raise AudioProcessingError(f"Failed to load audio file: {str(e)}")
    
    try:
        processed_audio = apply_audio_operations(
            audio=audio,
            audio_processor=audio_processor,
            operations=edit_request.operations,
            trim_silence=edit_request.trim_silence,
            normalize=edit_request.normalize,
            fade_in=edit_request.fade_in,
            fade_out=edit_request.fade_out
        )
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        raise AudioProcessingError(f"Failed to process audio: {str(e)}")
    
    new_song_id = generate_song_id()
    output_path = get_audio_file_path(new_song_id)
    
    try:
        save_audio_file(processed_audio, output_path, sample_rate)
    except Exception as e:
        logger.error(f"Error saving audio file: {e}", exc_info=True)
        raise AudioProcessingError(f"Failed to save audio file: {str(e)}")
    
    # Guardar usando servicio async si está disponible
    try:
        await get_song_async_or_sync(
            song_service,
            'save_song',
            song_id=new_song_id,
            user_id=song.get("user_id"),
            prompt=f"Edited version of {song_id}",
            file_path=str(output_path),
            metadata={"original_song_id": song_id, "operations": edit_request.operations}
        )
    except (AttributeError, TypeError):
        # Fallback a servicio síncrono si save_song no es async
        song_service.save_song(
            song_id=new_song_id,
            user_id=song.get("user_id"),
            prompt=f"Edited version of {song_id}",
            file_path=str(output_path),
            metadata={"original_song_id": song_id, "operations": edit_request.operations}
        )
    
    logger.info(f"Song {song_id} edited successfully, new ID: {new_song_id}")
    
    return SongResponse(
        song_id=new_song_id,
        status="completed",
        message="Canción editada exitosamente",
        audio_url=f"/suno/songs/{new_song_id}/download"
    )


@router.post("/mix", response_model=SongResponse)
async def mix_songs(
    mix_request: AudioMixRequest,
    song_service: SongServiceDep = Depends(),
    audio_processor: AudioProcessorDep = Depends()
) -> SongResponse:
    """
    Mezcla múltiples canciones en una sola pista de audio.
    
    Combina varias canciones sincronizándolas y aplicando volúmenes opcionales.
    """
    # Guard clause: validar que hay al menos una canción
    if not mix_request.song_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one song ID is required"
        )
    
    audio_tracks: List[Any] = []
    sample_rate: Optional[int] = None
    valid_song_ids: List[str] = []
    
    # Cargar y validar cada canción
    for song_id in mix_request.song_ids:
        # Validar formato UUID
        try:
            uuid.UUID(song_id)
        except ValueError:
            logger.warning(f"Invalid song ID format: {song_id}, skipping")
            continue
        
        # Verificar existencia de canción
        song = song_service.get_song(song_id)
        if not song:
            logger.warning(f"Song {song_id} not found, skipping")
            continue
        
        # Verificar existencia de archivo
        file_path = Path(song.get("file_path", ""))
        if not file_path.exists():
            logger.warning(f"Audio file for {song_id} not found, skipping")
            continue
        
        # Cargar audio
        try:
            audio, sr = load_audio_file(str(file_path))
            audio_tracks.append(audio)
            if sample_rate is None:
                sample_rate = sr
            valid_song_ids.append(song_id)
        except Exception as read_error:
            logger.warning(f"Error reading {song_id}: {read_error}, skipping")
            continue
    
    # Guard clause: verificar que hay canciones válidas
    if not audio_tracks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid songs found to mix"
        )
    
    if sample_rate is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not determine sample rate"
        )
    
    # Mezclar audio
    try:
        mixed_audio = audio_processor.mix_audio(
            audio_tracks,
            volumes=mix_request.volumes
        )
    except Exception as mix_error:
        logger.error(f"Error mixing audio: {mix_error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error mixing audio: {str(mix_error)}"
        )
    
    # Guardar canción mezclada
    new_song_id = generate_song_id()
    output_path = get_audio_file_path(new_song_id)
    save_audio_file(mixed_audio, output_path, sample_rate)
    
    song_service.save_song(
        song_id=new_song_id,
        user_id=None,
        prompt=f"Mixed from {len(valid_song_ids)} songs",
        file_path=str(output_path),
        metadata={
            "mixed_songs": valid_song_ids,
            "original_count": len(mix_request.song_ids),
            "valid_count": len(valid_song_ids),
            "volumes": mix_request.volumes
        }
    )
    
    logger.info(f"Mixed {len(valid_song_ids)} songs successfully, new ID: {new_song_id}")
    
    return SongResponse(
        song_id=new_song_id,
        status="completed",
        message=f"Se mezclaron {len(valid_song_ids)} canciones exitosamente",
        audio_url=f"/suno/songs/{new_song_id}/download"
    )


@router.get("/{song_id}/analyze", response_model=SongAnalysisResponse)
@cache_response(ttl=300)  # Cache por 5 minutos (análisis no cambia)
async def analyze_song(
    song_id: str,
    response: Optional[Response] = None,
    song_service: SongServiceDep = Depends(),
    audio_processor: AudioProcessorDep = Depends()
) -> SongAnalysisResponse:
    """
    Analiza características de una canción (optimizado).
    
    Usa caching ya que el análisis no cambia para la misma canción.
    """
    validate_song_id(song_id)
    
    # Obtener canción usando servicio async
    song = await get_song_async_or_sync(song_service, 'get_song', song_id)
    song = ensure_song_exists(song, song_id)
    
    # Guard clause: verificar file_path
    file_path_str = song.get("file_path", "")
    if not file_path_str:
        raise AudioProcessingError(f"File path not found for song {song_id}")
    
    try:
        audio, sample_rate = load_audio_file(file_path_str)
        analysis = audio_processor.analyze_audio(audio)
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}", exc_info=True)
        raise AudioProcessingError(f"Failed to analyze audio: {str(e)}")
    
    # Headers de cache
    if response:
        response.headers["Cache-Control"] = "public, max-age=300"
    
    return SongAnalysisResponse(
        song_id=song_id,
        analysis=analysis,
        metadata=song.get("metadata", {})
    )

