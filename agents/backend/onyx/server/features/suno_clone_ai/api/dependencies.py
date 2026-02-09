"""
Dependencies para inyección en FastAPI

Sigue el patrón de dependency injection de FastAPI para gestionar
recursos compartidos y servicios de forma eficiente.

Usa funciones async donde sea posible para mejor rendimiento.
"""

from typing import Annotated, Optional, TYPE_CHECKING
from functools import lru_cache
from fastapi import Depends

if TYPE_CHECKING:
    from ..services.song_service import SongService
    from ..services.song_service_async import SongServiceAsync
    from ..core.music_generator import MusicGenerator
    from ..core.chat_processor import ChatProcessor
    from ..core.cache_manager import CacheManager
    from ..core.audio_processor import AudioProcessor
    from ..services.metrics_service import MetricsService
    from ..services.notification_service import NotificationService

from ..services.song_service import SongService
from ..core.music_generator import get_music_generator
from ..core.chat_processor import get_chat_processor
from ..core.cache_manager import get_cache_manager
from ..core.audio_processor import get_audio_processor
from ..services.metrics_service import get_metrics_service
from ..services.notification_service import get_notification_service


# Cache de instancias para evitar recrear servicios
_song_service_instance: Optional[SongService] = None


@lru_cache(maxsize=1)
def get_song_service() -> SongService:
    """
    Dependency para obtener el servicio de canciones.
    
    Usa lru_cache para reutilizar la instancia.
    """
    global _song_service_instance
    if _song_service_instance is None:
        _song_service_instance = SongService()
    return _song_service_instance


async def get_song_service_async() -> "SongServiceAsync":
    """
    Dependency async para obtener el servicio de canciones async.
    
    Preferir este sobre get_song_service para mejor rendimiento.
    """
    try:
        from ..services.song_service_async import get_song_service_async as _get_async
        return await _get_async()
    except ImportError:
        # Fallback a servicio síncrono si async no está disponible
        return get_song_service()


@lru_cache(maxsize=1)
def get_music_gen() -> "MusicGenerator":
    """Dependency para obtener el generador de música"""
    return get_music_generator()


@lru_cache(maxsize=1)
def get_chat_proc() -> "ChatProcessor":
    """Dependency para obtener el procesador de chat"""
    return get_chat_processor()


@lru_cache(maxsize=1)
def get_cache_mgr() -> "CacheManager":
    """Dependency para obtener el gestor de caché"""
    return get_cache_manager()


@lru_cache(maxsize=1)
def get_audio_proc() -> "AudioProcessor":
    """Dependency para obtener el procesador de audio"""
    return get_audio_processor()


@lru_cache(maxsize=1)
def get_metrics_svc() -> "MetricsService":
    """Dependency para obtener el servicio de métricas"""
    return get_metrics_service()


def get_notification_svc() -> Optional["NotificationService"]:
    """
    Dependency para obtener el servicio de notificaciones.
    
    Returns None si el servicio no está disponible (no crítico).
    Usa early return pattern para mejor legibilidad.
    """
    try:
        return get_notification_service()
    except Exception:
        return None


# Type aliases para dependencies usando Annotated
SongServiceDep = Annotated[SongService, Depends(get_song_service)]
SongServiceAsyncDep = Annotated["SongServiceAsync", Depends(get_song_service_async)]
MusicGeneratorDep = Annotated["MusicGenerator", Depends(get_music_gen)]
ChatProcessorDep = Annotated["ChatProcessor", Depends(get_chat_proc)]
CacheManagerDep = Annotated["CacheManager", Depends(get_cache_mgr)]
AudioProcessorDep = Annotated["AudioProcessor", Depends(get_audio_proc)]
MetricsServiceDep = Annotated["MetricsService", Depends(get_metrics_svc)]
NotificationServiceDep = Annotated[Optional["NotificationService"], Depends(get_notification_svc)]

