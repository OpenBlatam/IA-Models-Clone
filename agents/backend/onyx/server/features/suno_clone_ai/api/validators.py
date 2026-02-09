"""
Validadores para endpoints

Funciones puras que validan inputs y usan excepciones personalizadas
siguiendo el patrón de early returns y guard clauses.
"""

import uuid
from typing import Optional
from pathlib import Path

from .exceptions import (
    InvalidInputError,
    SongNotFoundError,
    AudioFileNotFoundError
)


def validate_song_id(song_id: str) -> str:
    """
    Valida que el song_id sea un UUID válido.
    
    Args:
        song_id: ID de la canción a validar
        
    Returns:
        El song_id si es válido
        
    Raises:
        InvalidInputError: Si el formato del ID es inválido
    """
    # Guard clause: validación temprana
    if not song_id or not isinstance(song_id, str):
        raise InvalidInputError("Song ID must be a non-empty string")
    
    try:
        uuid.UUID(song_id)
        return song_id
    except ValueError:
        raise InvalidInputError(f"Invalid song ID format: {song_id}")


def ensure_song_exists(song: Optional[dict], song_id: str) -> dict:
    """
    Asegura que la canción existe.
    
    Args:
        song: Diccionario de la canción (puede ser None)
        song_id: ID de la canción
        
    Returns:
        El diccionario de la canción si existe
        
    Raises:
        SongNotFoundError: Si la canción no existe
    """
    # Guard clause: early return pattern
    if not song:
        raise SongNotFoundError(song_id)
    
    return song


def ensure_audio_file_exists(file_path: str, song_id: Optional[str] = None) -> None:
    """
    Asegura que el archivo de audio existe.
    
    Args:
        file_path: Ruta al archivo de audio
        song_id: ID de la canción (opcional, para mensajes de error)
        
    Raises:
        AudioFileNotFoundError: Si el archivo no existe
    """
    # Guard clause: validación temprana
    if not file_path:
        error_msg = "Audio file path is required"
        if song_id:
            error_msg += f" (song_id: {song_id})"
        raise AudioFileNotFoundError(error_msg)
    
    path = Path(file_path)
    if not path.exists():
        error_msg = str(file_path)
        if song_id:
            error_msg += f" (song_id: {song_id})"
        raise AudioFileNotFoundError(error_msg)
    
    if not path.is_file():
        error_msg = f"Path is not a file: {file_path}"
        if song_id:
            error_msg += f" (song_id: {song_id})"
        raise AudioFileNotFoundError(error_msg)

