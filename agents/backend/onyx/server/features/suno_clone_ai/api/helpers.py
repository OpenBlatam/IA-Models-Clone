"""
Funciones helper para operaciones comunes

Funciones puras y utilitarias que siguen el principio de responsabilidad única.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
import numpy as np
import soundfile as sf

try:
    from config.settings import settings
except ImportError:
    from ..config.settings import settings

if TYPE_CHECKING:
    from ..core.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


def generate_song_id() -> str:
    """Genera un ID único para una canción"""
    return str(uuid.uuid4())


def ensure_storage_dir() -> Path:
    """Asegura que el directorio de almacenamiento existe"""
    storage_path = Path(settings.audio_storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path


def get_audio_file_path(song_id: str, extension: str = "wav") -> Path:
    """Obtiene la ruta del archivo de audio para una canción"""
    storage_dir = ensure_storage_dir()
    return storage_dir / f"{song_id}.{extension}"


def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Carga un archivo de audio.
    
    Args:
        file_path: Ruta al archivo de audio
        
    Returns:
        Tupla con (audio_data, sample_rate)
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo no es válido
    """
    # Guard clause: verificar que el archivo existe
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    try:
        return sf.read(str(path))
    except Exception as e:
        raise ValueError(f"Failed to load audio file {file_path}: {str(e)}") from e


def save_audio_file(audio: np.ndarray, file_path: Path, sample_rate: int) -> None:
    """
    Guarda un archivo de audio.
    
    Args:
        audio: Datos de audio como numpy array
        file_path: Ruta donde guardar el archivo
        sample_rate: Tasa de muestreo del audio
        
    Raises:
        ValueError: Si los parámetros son inválidos
        OSError: Si no se puede escribir el archivo
    """
    # Guard clauses: validación temprana
    if audio.size == 0:
        raise ValueError("Audio data is empty")
    
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")
    
    if not isinstance(file_path, Path):
        raise ValueError(f"file_path must be a Path object, got {type(file_path)}")
    
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(file_path), audio, sample_rate)
    except Exception as e:
        raise OSError(f"Failed to save audio file to {file_path}: {str(e)}") from e


def create_song_info_from_request(request: Any) -> Dict[str, Any]:
    """Crea información de canción desde un request"""
    return {
        "prompt": request.prompt,
        "genre": request.genre,
        "mood": request.mood,
        "duration": request.duration,
        "instruments": []
    }


async def notify_song_started(
    notification_service: Optional[Any],
    user_id: Optional[str],
    song_id: str
) -> None:
    """
    Notifica que la generación de una canción ha comenzado.
    
    Args:
        notification_service: Servicio de notificaciones (puede ser None)
        user_id: ID del usuario
        song_id: ID de la canción
    """
    # Early return si no hay servicio o usuario
    if not user_id or not notification_service:
        return
    
    try:
        await notification_service.notify_generation_started(user_id, song_id)
    except Exception as e:
        logger.warning(f"Could not send start notification: {e}")


async def notify_song_completed(
    notification_service: Optional[Any],
    user_id: Optional[str],
    song_id: str,
    audio_url: str
) -> None:
    """
    Notifica que una canción ha sido completada.
    
    Args:
        notification_service: Servicio de notificaciones (puede ser None)
        user_id: ID del usuario
        song_id: ID de la canción
        audio_url: URL del audio generado
    """
    # Early return si no hay servicio o usuario
    if not user_id or not notification_service:
        return
    
    try:
        await notification_service.notify_song_completed(user_id, song_id, audio_url)
    except Exception as e:
        logger.warning(f"Could not send completion notification: {e}")


async def notify_song_failed(
    notification_service: Optional[Any],
    user_id: Optional[str],
    song_id: str,
    error_message: str
) -> None:
    """
    Notifica que la generación de una canción falló.
    
    Args:
        notification_service: Servicio de notificaciones (puede ser None)
        user_id: ID del usuario
        song_id: ID de la canción
        error_message: Mensaje de error
    """
    # Early return si no hay servicio o usuario
    if not user_id or not notification_service:
        return
    
    try:
        await notification_service.notify_song_failed(user_id, song_id, error_message)
    except Exception as e:
        logger.warning(f"Could not send failure notification: {e}")


def apply_audio_operations(
    audio: np.ndarray,
    audio_processor: "AudioProcessor",
    operations: List[Dict[str, Any]],
    trim_silence: bool = False,
    normalize: bool = True,
    fade_in: Optional[float] = None,
    fade_out: Optional[float] = None
) -> np.ndarray:
    """
    Aplica operaciones de procesamiento de audio en secuencia.
    
    Args:
        audio: Datos de audio como numpy array
        audio_processor: Instancia del procesador de audio
        operations: Lista de operaciones a aplicar
        trim_silence: Si eliminar silencio al inicio/final
        normalize: Si normalizar el audio
        fade_in: Segundos de fade in (None para desactivar)
        fade_out: Segundos de fade out (None para desactivar)
        
    Returns:
        Audio procesado como numpy array
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    # Guard clauses: validación temprana
    if audio.size == 0:
        raise ValueError("Audio data is empty")
    
    if not operations:
        operations = []
    
    # Aplicar operaciones básicas primero
    if trim_silence:
        audio = audio_processor.trim_silence(audio)
    
    if normalize:
        audio = audio_processor.normalize(audio)
    
    if fade_in is not None or fade_out is not None:
        audio = audio_processor.apply_fade(
            audio,
            fade_in=fade_in or 0.0,
            fade_out=fade_out or 0.0
        )
    
    # Aplicar operaciones personalizadas
    for op in operations:
        op_type = op.get("type")
        if not op_type:
            continue  # Skip invalid operations
        
        if op_type == "reverb":
            audio = audio_processor.apply_reverb(
                audio,
                room_size=op.get("room_size", 0.5),
                damping=op.get("damping", 0.5)
            )
        elif op_type == "eq":
            audio = audio_processor.apply_eq(
                audio,
                low_gain=op.get("low_gain", 0.0),
                mid_gain=op.get("mid_gain", 0.0),
                high_gain=op.get("high_gain", 0.0)
            )
        elif op_type == "tempo":
            audio = audio_processor.change_tempo(
                audio,
                factor=op.get("factor", 1.0)
            )
        elif op_type == "pitch":
            audio = audio_processor.change_pitch(
                audio,
                semitones=op.get("semitones", 0.0)
            )
    
    return audio

