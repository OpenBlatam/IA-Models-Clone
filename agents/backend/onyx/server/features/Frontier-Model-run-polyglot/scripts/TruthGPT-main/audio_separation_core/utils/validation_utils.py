"""
Validation Utilities - Utilidades de validación.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


def validate_audio_path(path: Union[str, Path]) -> Path:
    """
    Valida y normaliza una ruta de audio.
    
    Args:
        path: Ruta a validar
    
    Returns:
        Path validado
    
    Raises:
        ValueError: Si la ruta no es válida
    """
    path = Path(path)
    
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    return path.resolve()


def validate_output_dir(path: Union[str, Path], create: bool = True) -> Path:
    """
    Valida y crea un directorio de salida.
    
    Args:
        path: Ruta al directorio
        create: Si True, crea el directorio si no existe
    
    Returns:
        Path validado
    
    Raises:
        ValueError: Si la ruta no es válida
    """
    path = Path(path)
    
    if path.exists() and not path.is_dir():
        raise ValueError(f"Path exists but is not a directory: {path}")
    
    if create:
        path.mkdir(parents=True, exist_ok=True)
    
    return path.resolve()


def validate_volume(volume: float, name: str = "volume") -> None:
    """
    Valida que un volumen esté en el rango válido.
    
    Args:
        volume: Volumen a validar (0.0-1.0)
        name: Nombre del parámetro para mensajes de error
    
    Raises:
        ValueError: Si el volumen no es válido
    """
    if not isinstance(volume, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(volume)}")
    
    if not 0.0 <= volume <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {volume}")




