"""
Helpers generales para tests
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import soundfile as sf


def create_mock_audio(
    duration: float = 1.0,
    sample_rate: int = 44100,
    frequency: float = 440.0,
    channels: int = 1
) -> np.ndarray:
    """
    Crea audio mock para testing
    
    Args:
        duration: Duración en segundos
        sample_rate: Sample rate
        frequency: Frecuencia en Hz
        channels: Número de canales
    
    Returns:
        Array de audio
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    if channels > 1:
        audio = np.stack([audio] * channels, axis=1)
    
    return audio.astype(np.float32)


def save_test_audio(
    audio: np.ndarray,
    file_path: Path,
    sample_rate: int = 44100
) -> Path:
    """
    Guarda audio de prueba en un archivo
    
    Args:
        audio: Array de audio
        file_path: Ruta del archivo
        sample_rate: Sample rate
    
    Returns:
        Path del archivo guardado
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(file_path), audio, sample_rate)
    return file_path


def load_test_audio(file_path: Path) -> tuple:
    """
    Carga audio de prueba desde un archivo
    
    Args:
        file_path: Ruta del archivo
    
    Returns:
        Tuple de (audio, sample_rate)
    """
    return sf.read(str(file_path))


def create_song_dict(
    song_id: str = "test-song-123",
    user_id: str = "test-user-456",
    prompt: str = "Test song",
    status: str = "completed",
    **kwargs
) -> Dict[str, Any]:
    """
    Crea un diccionario de canción para testing
    
    Args:
        song_id: ID de la canción
        user_id: ID del usuario
        prompt: Prompt de generación
        status: Estado de la canción
        **kwargs: Campos adicionales
    
    Returns:
        Diccionario de canción
    """
    return {
        "song_id": song_id,
        "user_id": user_id,
        "prompt": prompt,
        "status": status,
        "file_path": f"/tmp/{song_id}.wav",
        "metadata": {},
        **kwargs
    }


def assert_song_dict_valid(song: Dict[str, Any]) -> None:
    """
    Verifica que un diccionario de canción sea válido
    
    Args:
        song: Diccionario de canción
    """
    assert "song_id" in song, "Missing song_id"
    assert "status" in song, "Missing status"
    assert isinstance(song["song_id"], str), "song_id must be string"
    assert isinstance(song["status"], str), "status must be string"


def assert_audio_valid(
    audio: np.ndarray,
    min_length: int = 1,
    max_length: Optional[int] = None
) -> None:
    """
    Verifica que audio sea válido
    
    Args:
        audio: Array de audio
        min_length: Longitud mínima esperada
        max_length: Longitud máxima esperada
    """
    assert audio is not None, "Audio is None"
    assert isinstance(audio, np.ndarray), "Audio must be numpy array"
    assert len(audio) >= min_length, f"Audio too short: {len(audio)} < {min_length}"
    
    if max_length:
        assert len(audio) <= max_length, f"Audio too long: {len(audio)} > {max_length}"


def generate_test_song_id(prefix: str = "test") -> str:
    """
    Genera un ID de canción para testing
    
    Args:
        prefix: Prefijo para el ID
    
    Returns:
        ID generado
    """
    import uuid
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

