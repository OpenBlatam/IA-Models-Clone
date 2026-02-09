"""
Helpers y Utilidades Comunes

Funciones helper reutilizables para el sistema.
"""

import logging
import hashlib
import json
from typing import Any, Dict, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_id(prefix: str = "") -> str:
    """
    Genera un ID único
    
    Args:
        prefix: Prefijo opcional
    
    Returns:
        ID único
    """
    import uuid
    id_str = str(uuid.uuid4())
    return f"{prefix}_{id_str}" if prefix else id_str


def hash_string(value: str, algorithm: str = "sha256") -> str:
    """
    Hashea un string
    
    Args:
        value: Valor a hashear
        algorithm: Algoritmo (md5, sha1, sha256)
    
    Returns:
        Hash hexadecimal
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(value.encode())
    return hash_obj.hexdigest()


def safe_json_loads(value: str, default: Any = None) -> Any:
    """
    Carga JSON de forma segura
    
    Args:
        value: String JSON
        default: Valor por defecto si falla
    
    Returns:
        Objeto parseado o default
    """
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(value: Any, default: str = "{}") -> str:
    """
    Serializa a JSON de forma segura
    
    Args:
        value: Objeto a serializar
        default: Valor por defecto si falla
    
    Returns:
        String JSON o default
    """
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return default


def format_duration(seconds: float) -> str:
    """
    Formatea duración en formato legible
    
    Args:
        seconds: Segundos
    
    Returns:
        String formateado (ej: "3:45")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def format_file_size(bytes_size: int) -> str:
    """
    Formatea tamaño de archivo
    
    Args:
        bytes_size: Tamaño en bytes
    
    Returns:
        String formateado (ej: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def ensure_directory(path: str) -> Path:
    """
    Asegura que un directorio existe
    
    Args:
        path: Ruta del directorio
    
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Divide una lista en chunks
    
    Args:
        items: Lista de items
        chunk_size: Tamaño del chunk
    
    Returns:
        Lista de chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusiona múltiples diccionarios
    
    Args:
        *dicts: Diccionarios a fusionar
    
    Returns:
        Diccionario fusionado
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def get_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Obtiene un valor anidado de un diccionario
    
    Args:
        data: Diccionario
        path: Ruta separada por puntos (ej: "user.profile.name")
        default: Valor por defecto
    
    Returns:
        Valor o default
    """
    keys = path.split('.')
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    return value


def set_nested_value(data: Dict[str, Any], path: str, value: Any):
    """
    Establece un valor anidado en un diccionario
    
    Args:
        data: Diccionario
        path: Ruta separada por puntos
        value: Valor a establecer
    """
    keys = path.split('.')
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def sanitize_filename(filename: str) -> str:
    """
    Sanitiza un nombre de archivo
    
    Args:
        filename: Nombre de archivo
    
    Returns:
        Nombre sanitizado
    """
    import re
    # Remover caracteres peligrosos
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limitar longitud
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255 - len(ext) - 1] + '.' + ext
    return filename


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorador para reintentar en caso de fallo
    
    Args:
        max_retries: Número máximo de reintentos
        delay: Delay entre reintentos (segundos)
    """
    import asyncio
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator

