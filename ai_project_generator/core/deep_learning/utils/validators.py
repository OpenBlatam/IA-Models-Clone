"""
Validators - Funciones de validación
=====================================

Funciones puras para validar inputs y parámetros.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

from pathlib import Path
from typing import Optional

from ..generator_config import GENERATOR_MAP


def validate_generator_key(generator_key: str) -> None:
    """
    Validar clave de generador (función pura).
    
    Args:
        generator_key: Clave del generador
        
    Raises:
        ValueError: Si la clave es inválida
    """
    if not generator_key or not generator_key.strip():
        raise ValueError("generator_key cannot be empty")


def validate_project_path(project_dir: Path) -> None:
    """
    Validar ruta del proyecto (función pura).
    
    Args:
        project_dir: Directorio del proyecto
        
    Raises:
        ValueError: Si la ruta es inválida
        TypeError: Si no es un Path
    """
    if project_dir is None:
        raise ValueError("project_dir cannot be None")
    
    if not isinstance(project_dir, Path):
        raise TypeError("project_dir must be a Path object")


def get_target_directory(
    project_dir: Path,
    generator_key: str
) -> Optional[Path]:
    """
    Obtener directorio objetivo para el generador (función pura).
    
    Args:
        project_dir: Directorio del proyecto
        generator_key: Clave del generador
        
    Returns:
        Directorio objetivo o None si no se encuentra configuración
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    if project_dir is None:
        raise ValueError("project_dir cannot be None")
    
    if not generator_key:
        raise ValueError("generator_key cannot be empty")
    
    if generator_key not in GENERATOR_MAP:
        return None
    
    _, _, subdir = GENERATOR_MAP[generator_key]
    return project_dir / "app" / subdir
