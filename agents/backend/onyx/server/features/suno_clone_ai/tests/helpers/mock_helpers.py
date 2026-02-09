"""
Helpers para crear mocks
"""

from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, Optional, List
import numpy as np


def create_mock_song_service(
    songs: Optional[List[Dict[str, Any]]] = None,
    default_song: Optional[Dict[str, Any]] = None
) -> Mock:
    """
    Crea un mock del servicio de canciones
    
    Args:
        songs: Lista de canciones para list_songs
        default_song: Canción por defecto para get_song
    
    Returns:
        Mock del SongService
    """
    service = Mock()
    
    # Configurar list_songs
    if songs is not None:
        service.list_songs = Mock(return_value=songs)
    else:
        service.list_songs = Mock(return_value=[])
    
    # Configurar get_song
    def get_song_side_effect(song_id: str):
        if default_song and default_song.get("song_id") == song_id:
            return default_song
        return None
    
    service.get_song = Mock(side_effect=get_song_side_effect)
    service.save_song = Mock(return_value=True)
    service.delete_song = Mock(return_value=True)
    service.update_song_status = Mock(return_value=True)
    service.get_chat_history = Mock(return_value=[])
    
    return service


def create_mock_music_generator(
    audio_output: Optional[np.ndarray] = None
) -> Mock:
    """
    Crea un mock del generador de música
    
    Args:
        audio_output: Audio a retornar en generate_from_text
    
    Returns:
        Mock del MusicGenerator
    """
    generator = Mock()
    
    if audio_output is None:
        audio_output = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    
    generator.generate_from_text = Mock(return_value=audio_output)
    generator.save_audio = Mock(return_value=True)
    
    return generator


def create_mock_audio_processor(
    passthrough: bool = True
) -> Mock:
    """
    Crea un mock del procesador de audio
    
    Args:
        passthrough: Si True, las operaciones retornan el audio sin modificar
    
    Returns:
        Mock del AudioProcessor
    """
    processor = Mock()
    
    if passthrough:
        # Todas las operaciones retornan el audio sin modificar
        processor.normalize = Mock(side_effect=lambda x: x)
        processor.apply_fade = Mock(side_effect=lambda x, **kwargs: x)
        processor.trim_silence = Mock(side_effect=lambda x: x)
        processor.apply_reverb = Mock(side_effect=lambda x, **kwargs: x)
        processor.apply_eq = Mock(side_effect=lambda x, **kwargs: x)
        processor.change_tempo = Mock(side_effect=lambda x, **kwargs: x)
        processor.change_pitch = Mock(side_effect=lambda x, **kwargs: x)
        processor.mix_audio = Mock(side_effect=lambda tracks, **kwargs: tracks[0] if tracks else np.array([]))
    else:
        # Implementaciones básicas
        processor.normalize = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        processor.apply_fade = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        processor.trim_silence = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        processor.apply_reverb = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        processor.apply_eq = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        processor.change_tempo = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        processor.change_pitch = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        processor.mix_audio = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    
    processor.analyze_audio = Mock(return_value={
        "duration": 30.0,
        "sample_rate": 44100,
        "channels": 2,
        "rms": 0.5,
        "peak": 1.0
    })
    
    return processor


def create_mock_notification_service() -> AsyncMock:
    """
    Crea un mock del servicio de notificaciones
    
    Returns:
        AsyncMock del NotificationService
    """
    service = AsyncMock()
    service.notify_generation_started = AsyncMock(return_value=True)
    service.notify_song_completed = AsyncMock(return_value=True)
    service.notify_song_failed = AsyncMock(return_value=True)
    return service


def create_mock_cache_manager(
    cache_data: Optional[Dict[str, Any]] = None
) -> Mock:
    """
    Crea un mock del gestor de caché
    
    Args:
        cache_data: Datos de caché predefinidos
    
    Returns:
        Mock del CacheManager
    """
    manager = Mock()
    
    if cache_data:
        def get_side_effect(*args, **kwargs):
            key = str(args) + str(kwargs)
            return cache_data.get(key)
        
        manager.get = Mock(side_effect=get_side_effect)
    else:
        manager.get = Mock(return_value=None)
    
    manager.set = Mock(return_value=True)
    manager.clear = Mock(return_value=True)
    manager.stats = Mock(return_value={
        "hits": 0,
        "misses": 0,
        "size": 0,
        "hit_rate": 0.0
    })
    
    return manager


def create_mock_chat_processor(
    song_info: Optional[Dict[str, Any]] = None
) -> Mock:
    """
    Crea un mock del procesador de chat
    
    Args:
        song_info: Información de canción a retornar
    
    Returns:
        Mock del ChatProcessor
    """
    processor = Mock()
    
    if song_info is None:
        song_info = {
            "prompt": "test song",
            "genre": "pop",
            "mood": "happy",
            "duration": 30
        }
    
    processor.extract_song_info = Mock(return_value=song_info)
    return processor


def create_mock_metrics_service() -> Mock:
    """
    Crea un mock del servicio de métricas
    
    Returns:
        Mock del MetricsService
    """
    service = Mock()
    service.record_generation = Mock(return_value=True)
    service.get_stats = Mock(return_value={
        "total_songs": 0,
        "total_users": 0,
        "avg_generation_time": 0.0,
        "success_rate": 1.0
    })
    service.get_user_stats = Mock(return_value={
        "user_id": "test_user",
        "songs_generated": 0,
        "avg_generation_time": 0.0
    })
    return service

