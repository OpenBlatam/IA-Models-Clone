"""
Lógica de negocio pura (funciones sin efectos secundarios)

Funciones puras que encapsulan la lógica de negocio sin efectos secundarios.
Siguen principios de programación funcional y son fácilmente testeables.
"""

import time
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path

try:
    from config.settings import settings
except ImportError:
    from ..config.settings import settings

if TYPE_CHECKING:
    from ..core.chat_processor import ChatProcessor
    from ..core.cache_manager import CacheManager
    from ..core.music_generator import MusicGenerator
    from ..core.audio_processor import AudioProcessor
    from ..services.song_service import SongService
    from ..services.metrics_service import MetricsService

logger = logging.getLogger(__name__)


def extract_song_info_from_chat(
    message: str,
    chat_history: Optional[List[Dict[str, Any]]],
    chat_processor: "ChatProcessor"
) -> Dict[str, Any]:
    """
    Extrae información de canción desde mensaje de chat.
    
    Args:
        message: Mensaje del usuario
        chat_history: Historial de conversación (opcional)
        chat_processor: Procesador de chat
        
    Returns:
        Diccionario con información extraída de la canción
        
    Raises:
        ValueError: Si el mensaje es inválido o no se puede procesar
    """
    # Guard clause: validar mensaje
    if not message or not isinstance(message, str):
        raise ValueError("Message must be a non-empty string")
    
    if not message.strip():
        raise ValueError("Message cannot be empty")
    
    try:
        return chat_processor.extract_song_info(message, chat_history or [])
    except Exception as e:
        logger.error(f"Error extracting song info from chat: {e}", exc_info=True)
        raise ValueError(f"Failed to extract song info: {str(e)}") from e


def prepare_song_generation(
    song_id: str,
    song_info: Dict[str, Any],
    user_id: Optional[str]
) -> Dict[str, Any]:
    """Prepara los datos para generación de canción"""
    return {
        "song_id": song_id,
        "song_info": song_info,
        "user_id": user_id,
        "start_time": time.time()
    }


def check_cache_for_song(
    cache_manager: "CacheManager",
    prompt: str,
    duration: Optional[int],
    genre: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Verifica si existe una canción en caché.
    
    Args:
        cache_manager: Gestor de caché
        prompt: Prompt de la canción
        duration: Duración en segundos (opcional)
        genre: Género musical (opcional)
        
    Returns:
        Diccionario con información de la canción en caché o None
    """
    # Guard clause: validar prompt
    if not prompt or not isinstance(prompt, str):
        return None
    
    try:
        cached_result = cache_manager.get(
            prompt=prompt,
            duration=duration,
            genre=genre
        )
        
        # Guard clause: verificar que existe resultado
        if not cached_result or not cached_result.get("file_path"):
            return None
        
        # Verificar que el archivo existe
        output_path = Path(cached_result["file_path"])
        if not output_path.exists():
            logger.warning(f"Cached file not found: {output_path}")
            return None
        
        return cached_result
    except Exception as e:
        logger.warning(f"Error checking cache: {e}")
        return None


def generate_and_process_audio(
    music_generator: "MusicGenerator",
    audio_processor: "AudioProcessor",
    prompt: str,
    duration: Optional[int]
) -> Any:
    """
    Genera y procesa audio con normalización y fade.
    
    Args:
        music_generator: Generador de música
        audio_processor: Procesador de audio
        prompt: Prompt para la generación
        duration: Duración en segundos (opcional)
        
    Returns:
        Audio procesado como numpy array
        
    Raises:
        ValueError: Si el prompt es inválido o la generación falla
    """
    # Guard clause: validar prompt
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")
    
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    try:
        # Generar audio
        audio = music_generator.generate_from_text(prompt, duration=duration)
        
        # Guard clause: verificar que se generó audio
        if audio is None:
            raise ValueError("Failed to generate audio")
        
        # Procesar audio
        audio = audio_processor.normalize(audio)
        audio = audio_processor.apply_fade(audio, fade_in=0.5, fade_out=0.5)
        
        return audio
    except Exception as e:
        logger.error(f"Error generating and processing audio: {e}", exc_info=True)
        raise ValueError(f"Failed to generate audio: {str(e)}") from e


def save_generated_song(
    song_service: "SongService",
    song_id: str,
    user_id: Optional[str],
    prompt: str,
    file_path: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Guarda la canción generada en la base de datos.
    
    Args:
        song_service: Servicio de canciones
        song_id: ID de la canción
        user_id: ID del usuario (opcional)
        prompt: Prompt usado para generar la canción
        file_path: Ruta al archivo de audio
        metadata: Metadatos adicionales
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    # Guard clauses: validación temprana
    if not song_id or not isinstance(song_id, str):
        raise ValueError("song_id must be a non-empty string")
    
    if not prompt or not isinstance(prompt, str):
        raise ValueError("prompt must be a non-empty string")
    
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path must be a non-empty string")
    
    # Verificar que el archivo existe
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Audio file not found: {file_path}")
    
    try:
        song_service.save_song(
            song_id=song_id,
            user_id=user_id,
            prompt=prompt,
            file_path=file_path,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Error saving song: {e}", exc_info=True)
        raise ValueError(f"Failed to save song: {str(e)}") from e


def record_generation_metrics(
    metrics_service: "MetricsService",
    song_id: str,
    user_id: Optional[str],
    prompt: str,
    duration: int,
    generation_time: float,
    status: str
) -> None:
    """
    Registra métricas de generación.
    
    Args:
        metrics_service: Servicio de métricas
        song_id: ID de la canción
        user_id: ID del usuario (opcional)
        prompt: Prompt usado
        duration: Duración en segundos
        generation_time: Tiempo de generación en segundos
        status: Estado de la generación (completed, failed, etc.)
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    # Guard clauses: validación temprana
    if not song_id:
        raise ValueError("song_id cannot be empty")
    
    if not isinstance(duration, int) or duration <= 0:
        raise ValueError(f"duration must be a positive integer, got: {duration}")
    
    if not isinstance(generation_time, (int, float)) or generation_time < 0:
        raise ValueError(f"generation_time must be non-negative, got: {generation_time}")
    
    if status not in ["completed", "failed", "processing", "cancelled"]:
        raise ValueError(f"Invalid status: {status}")
    
    try:
        metrics_service.record_generation(
            song_id=song_id,
            user_id=user_id,
            prompt=prompt,
            duration=duration,
            generation_time=generation_time,
            model_used=settings.music_model,
            status=status
        )
    except Exception as e:
        # Log pero no fallar si las métricas fallan
        logger.warning(f"Failed to record metrics: {e}")


def get_media_type_from_path(file_path: Path) -> str:
    """
    Determina el tipo MIME basado en la extensión del archivo.
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Tipo MIME del archivo (default: audio/wav)
    """
    # Guard clause: validar path
    if not file_path or not isinstance(file_path, Path):
        return "audio/wav"
    
    # Mapeo de extensiones a tipos MIME
    media_types: Dict[str, str] = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".wma": "audio/x-ms-wma",
        ".opus": "audio/opus"
    }
    
    suffix = file_path.suffix.lower()
    return media_types.get(suffix, "audio/wav")

