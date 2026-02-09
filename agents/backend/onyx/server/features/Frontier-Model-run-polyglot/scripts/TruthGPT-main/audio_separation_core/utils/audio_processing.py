"""
Audio Processing Utilities - Utilidades compartidas para procesamiento de audio.

Consolida imports y funciones comunes para evitar duplicación.
"""

from __future__ import annotations

from typing import Tuple, Optional
from pathlib import Path


def ensure_audio_libs() -> Tuple[any, any, any]:
    """
    Asegura que las librerías de audio estén disponibles.
    
    Returns:
        Tupla de (librosa, soundfile, numpy)
    
    Raises:
        ImportError: Si alguna librería no está disponible
    """
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        return librosa, sf, np
    except ImportError as e:
        raise ImportError(
            "Required audio libraries not installed. "
            "Install with: pip install librosa soundfile numpy"
        ) from e


def load_audio_mono(
    path: Path,
    sample_rate: Optional[int] = None,
    librosa
) -> Tuple[any, int]:
    """
    Carga un archivo de audio y lo convierte a mono.
    
    Args:
        path: Ruta al archivo de audio
        sample_rate: Sample rate objetivo (None = usar el del archivo)
        librosa: Módulo librosa importado
    
    Returns:
        Tupla de (audio array, sample_rate)
    """
    y, sr = librosa.load(str(path), sr=sample_rate, mono=False)
    
    # Convertir a mono si es necesario
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
    
    return y, sr


def normalize_audio(
    audio: any,
    headroom: float = 0.95,
    np
) -> any:
    """
    Normaliza un audio manteniendo headroom.
    
    Args:
        audio: Array de audio
        headroom: Factor de headroom (0.0-1.0)
        np: Módulo numpy importado
    
    Returns:
        Audio normalizado
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val * headroom
    return audio


def apply_fade(
    audio: any,
    sample_rate: int,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    np
) -> any:
    """
    Aplica fade in y fade out a un audio.
    
    Args:
        audio: Array de audio
        sample_rate: Sample rate
        fade_in: Segundos de fade in
        fade_out: Segundos de fade out
        np: Módulo numpy importado
    
    Returns:
        Audio con fade aplicado
    """
    processed = audio.copy()
    
    # Fade in
    if fade_in > 0:
        fade_samples = int(fade_in * sample_rate)
        if fade_samples < len(processed):
            fade_curve = np.linspace(0, 1, fade_samples)
            processed[:fade_samples] = processed[:fade_samples] * fade_curve
    
    # Fade out
    if fade_out > 0:
        fade_samples = int(fade_out * sample_rate)
        if fade_samples < len(processed):
            fade_curve = np.linspace(1, 0, fade_samples)
            processed[-fade_samples:] = processed[-fade_samples:] * fade_curve
    
    return processed

