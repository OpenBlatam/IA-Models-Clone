"""
Format Utilities - Utilidades para formatos de audio.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union


SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]


def is_audio_file(path: Union[str, Path]) -> bool:
    """
    Verifica si un archivo es de audio.
    
    Args:
        path: Ruta al archivo
    
    Returns:
        True si es un archivo de audio
    """
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_FORMATS


def is_video_file(path: Union[str, Path]) -> bool:
    """
    Verifica si un archivo es de video.
    
    Args:
        path: Ruta al archivo
    
    Returns:
        True si es un archivo de video
    """
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_FORMATS


def get_format_from_path(path: Union[str, Path]) -> str:
    """
    Obtiene el formato de un archivo desde su extensión.
    
    Args:
        path: Ruta al archivo
    
    Returns:
        Formato del archivo (sin punto)
    """
    return Path(path).suffix.lower().lstrip(".")


def validate_format(format: str, supported: List[str]) -> bool:
    """
    Valida que un formato esté soportado.
    
    Args:
        format: Formato a validar
        supported: Lista de formatos soportados
    
    Returns:
        True si el formato está soportado
    """
    format = format.lower().lstrip(".")
    return format in [f.lstrip(".") for f in supported]




