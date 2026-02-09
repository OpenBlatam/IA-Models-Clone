"""
Funciones helper para tareas en background
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from .helpers import (
    generate_song_id,
    get_audio_file_path,
    notify_song_started,
    notify_song_completed,
    notify_song_failed
)
from .business_logic import (
    check_cache_for_song,
    generate_and_process_audio,
    save_generated_song,
    record_generation_metrics
)
from ..config.settings import settings

logger = logging.getLogger(__name__)


async def _handle_cached_song(
    song_id: str,
    user_id: Optional[str],
    song_info: Dict[str, Any],
    cached_result: Dict[str, Any],
    song_service: Any,
    notification_service: Optional[Any]
) -> None:
    """
    Maneja el caso cuando la canción está en caché.
    
    Optimizado para usar servicio async si está disponible.
    """
    output_path = Path(cached_result["file_path"])
    logger.info(f"Using cached result for song: {song_id}")
    
    # Guard clause: verificar que el archivo existe
    if not output_path.exists():
        logger.warning(f"Cached file not found: {output_path}, will regenerate")
        return None
    
    # Guardar usando servicio async si está disponible
    try:
        from ..helpers.service_helpers import get_song_async_or_sync
        await get_song_async_or_sync(
            song_service,
            'save_song',
            song_id=song_id,
            user_id=user_id,
            prompt=song_info["prompt"],
            file_path=str(output_path),
            metadata={**song_info, "cached": True}
        )
    except (AttributeError, TypeError):
        # Fallback a servicio síncrono
        save_generated_song(
            song_service,
            song_id,
            user_id,
            song_info["prompt"],
            str(output_path),
            {**song_info, "cached": True}
        )
    
    await notify_song_completed(
        notification_service,
        user_id,
        song_id,
        f"/suno/songs/{song_id}/download"
    )


async def _generate_new_song(
    song_id: str,
    user_id: Optional[str],
    song_info: Dict[str, Any],
    start_time: float,
    song_service: Any,
    music_generator: Any,
    audio_processor: Any,
    cache_manager: Any,
    notification_service: Optional[Any]
) -> None:
    """Genera una nueva canción desde cero"""
    # Generar y procesar audio
    audio = generate_and_process_audio(
        music_generator,
        audio_processor,
        song_info["prompt"],
        song_info.get("duration")
    )
    
    # Guardar audio
    output_path = get_audio_file_path(song_id)
    music_generator.save_audio(audio, str(output_path))
    
    # Guardar en caché
    cache_manager.set(
        prompt=song_info["prompt"],
        result={"file_path": str(output_path)},
        duration=song_info.get("duration"),
        genre=song_info.get("genre")
    )
    
    # Guardar en base de datos usando servicio async si está disponible
    try:
        from ..helpers.service_helpers import get_song_async_or_sync
        await get_song_async_or_sync(
            song_service,
            'save_song',
            song_id=song_id,
            user_id=user_id,
            prompt=song_info["prompt"],
            file_path=str(output_path),
            metadata=song_info
        )
    except (AttributeError, TypeError):
        # Fallback a servicio síncrono
        save_generated_song(
            song_service,
            song_id,
            user_id,
            song_info["prompt"],
            str(output_path),
            song_info
        )
    
    # Registrar métricas
    generation_time = time.time() - start_time
    from ..services.metrics_service import get_metrics_service
    metrics_service = get_metrics_service()
    record_generation_metrics(
        metrics_service,
        song_id,
        user_id,
        song_info["prompt"],
        song_info.get("duration", settings.default_duration),
        generation_time,
        "completed"
    )
    
    logger.info(f"Song generated successfully: {song_id} in {generation_time:.2f}s")
    
    # Notificar completado
    await notify_song_completed(
        notification_service,
        user_id,
        song_id,
        f"/suno/songs/{song_id}/download"
    )


async def _handle_generation_error(
    song_id: str,
    user_id: Optional[str],
    song_info: Dict[str, Any],
    start_time: float,
    error: Exception,
    song_service: Any,
    notification_service: Optional[Any]
) -> None:
    """Maneja errores durante la generación"""
    logger.error(f"Error in background song generation: {error}", exc_info=True)
    song_service.update_song_status(song_id, "failed", str(error))
    
    # Registrar métricas de error
    generation_time = time.time() - start_time
    try:
        from ..services.metrics_service import get_metrics_service
        metrics_service = get_metrics_service()
        record_generation_metrics(
            metrics_service,
            song_id,
            user_id,
            song_info.get("prompt", ""),
            song_info.get("duration", 0),
            generation_time,
            "failed"
        )
    except Exception:
        logger.warning("Could not record error metrics")
    
    await notify_song_failed(notification_service, user_id, song_id, str(error))


async def generate_song_background(
    song_id: str,
    song_info: Dict[str, Any],
    user_id: Optional[str]
) -> None:
    """Genera la canción en background con notificaciones y métricas"""
    from ..core.music_generator import get_music_generator
    from ..core.cache_manager import get_cache_manager
    from ..core.audio_processor import get_audio_processor
    from ..services.song_service import SongService
    from ..services.notification_service import get_notification_service
    
    start_time = time.time()
    song_service = SongService()
    notification_service = get_notification_service()
    
    # Notificar inicio
    await notify_song_started(notification_service, user_id, song_id)
    
    try:
        # Verificar caché
        cache_manager = get_cache_manager()
        cached_result = check_cache_for_song(
            cache_manager,
            song_info["prompt"],
            song_info.get("duration"),
            song_info.get("genre")
        )
        
        # Early return si está en caché
        if cached_result:
            await _handle_cached_song(
                song_id,
                user_id,
                song_info,
                cached_result,
                song_service,
                notification_service
            )
            return
        
        # Generar nueva canción
        music_generator = get_music_generator()
        audio_processor = get_audio_processor(sample_rate=settings.sample_rate)
        
        await _generate_new_song(
            song_id,
            user_id,
            song_info,
            start_time,
            song_service,
            music_generator,
            audio_processor,
            cache_manager,
            notification_service
        )
        
    except Exception as e:
        await _handle_generation_error(
            song_id,
            user_id,
            song_info,
            start_time,
            e,
            song_service,
            notification_service
        )

