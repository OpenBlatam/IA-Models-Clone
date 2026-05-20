"""
Shared Utilities - Utilidades compartidas
==========================================

Módulo con utilidades compartidas entre diferentes componentes del generador.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import structlog

logger = structlog.get_logger(__name__)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Obtiene un logger estructurado usando structlog.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger estructurado configurado
    """
    return structlog.get_logger(name)


def validate_path(path: Path, must_exist: bool = False) -> None:
    """
    Valida una ruta (función pura).
    
    Args:
        path: Ruta a validar
        must_exist: Si True, la ruta debe existir
        
    Raises:
        ValueError: Si la ruta es inválida
    """
    if path is None:
        raise ValueError("path cannot be None")
    
    if must_exist and not path.exists():
        raise ValueError(f"path does not exist: {path}")


def ensure_directory(path: Path) -> Path:
    """
    Asegura que un directorio existe, creándolo si es necesario (función pura).
    
    Args:
        path: Ruta del directorio
        
    Returns:
        Path del directorio
    """
    if path is None:
        raise ValueError("path cannot be None")
    
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_write_file(
    file_path: Path,
    content: str,
    encoding: str = "utf-8",
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Escribe un archivo de forma segura.
    
    Args:
        file_path: Ruta del archivo
        content: Contenido a escribir
        encoding: Codificación del archivo
        logger_instance: Logger para errores (opcional)
        
    Raises:
        ValueError: Si los parámetros son inválidos
        IOError: Si no se puede escribir el archivo
    """
    if file_path is None:
        raise ValueError("file_path cannot be None")
    
    if content is None:
        raise ValueError("content cannot be None")
    
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
    except IOError as e:
        error_msg = f"Failed to write file {file_path}: {e}"
        if logger_instance:
            logger_instance.error(error_msg)
        raise IOError(error_msg) from e


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """
    Sanitiza un nombre para usarlo como nombre de archivo (función pura).
    
    Args:
        name: Nombre a sanitizar
        max_length: Longitud máxima
        
    Returns:
        Nombre sanitizado
    """
    if not name:
        return "unnamed"
    
    if max_length < 1:
        raise ValueError("max_length must be positive")
    
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s-]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'-+', '_', name)
    name = name.strip('_')
    return name[:max_length]


def format_project_name(name: str) -> str:
    """
    Formatea un nombre de proyecto para mostrar (función pura).
    
    Args:
        name: Nombre del proyecto
        
    Returns:
        Nombre formateado
    """
    if not name:
        return ""
    
    return name.replace('_', ' ').title()


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusiona múltiples diccionarios (función pura).
    
    Args:
        *dicts: Diccionarios a fusionar
        
    Returns:
        Diccionario fusionado
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def get_nested_value(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Obtiene un valor anidado de un diccionario usando una ruta de claves (función pura).
    
    Args:
        data: Diccionario
        key_path: Ruta de claves separadas por puntos (ej: "user.profile.name")
        default: Valor por defecto
        
    Returns:
        Valor encontrado o default
    """
    if not data or not key_path:
        return default
    
    keys = key_path.split('.')
    value = data
    
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    
    return value if value is not None else default


def set_nested_value(data: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Establece un valor anidado en un diccionario usando una ruta de claves.
    
    Args:
        data: Diccionario
        key_path: Ruta de claves separadas por puntos
        value: Valor a establecer
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    if data is None:
        raise ValueError("data cannot be None")
    
    if not key_path:
        raise ValueError("key_path cannot be empty")
    
    keys = key_path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
