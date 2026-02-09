"""
Environment Utilities - Utilidades de entorno
==============================================

Utilidades para trabajar con variables de entorno y configuración.
"""

import os
from typing import Optional, Any, Dict, Union
from pathlib import Path


def get_env(
    key: str,
    default: Optional[str] = None,
    required: bool = False
) -> Optional[str]:
    """
    Obtener variable de entorno.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
        required: Si True, lanza error si no existe
    
    Returns:
        Valor de la variable
    
    Raises:
        ValueError: Si required=True y la variable no existe
    """
    value = os.environ.get(key, default)
    
    if value is None and required:
        raise ValueError(f"Required environment variable {key} not set")
    
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Obtener variable booleana de entorno.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
    
    Returns:
        Valor booleano
    """
    value = os.environ.get(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on', 'enabled')


def get_env_int(key: str, default: int = 0) -> int:
    """
    Obtener variable entera de entorno.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
    
    Returns:
        Valor entero
    """
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Obtener variable flotante de entorno.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
    
    Returns:
        Valor flotante
    """
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def get_env_list(
    key: str,
    separator: str = ',',
    default: Optional[list] = None
) -> list:
    """
    Obtener variable como lista.
    
    Args:
        key: Nombre de la variable
        separator: Separador
        default: Valor por defecto
    
    Returns:
        Lista
    """
    value = os.environ.get(key)
    if value is None:
        return default or []
    return [item.strip() for item in value.split(separator) if item.strip()]


def set_env(key: str, value: Any, override: bool = True):
    """
    Establecer variable de entorno.
    
    Args:
        key: Nombre de la variable
        value: Valor
        override: Si True, sobrescribe si existe
    """
    if key in os.environ and not override:
        return
    os.environ[key] = str(value)


def unset_env(key: str):
    """
    Eliminar variable de entorno.
    
    Args:
        key: Nombre de la variable
    """
    os.environ.pop(key, None)


def has_env(key: str) -> bool:
    """
    Verificar si variable existe.
    
    Args:
        key: Nombre de la variable
    
    Returns:
        True si existe
    """
    return key in os.environ


def load_env_file(
    file_path: Union[str, Path],
    override: bool = True
) -> Dict[str, str]:
    """
    Cargar variables desde archivo .env.
    
    Args:
        file_path: Ruta al archivo
        override: Si True, sobrescribe variables existentes
    
    Returns:
        Diccionario con variables cargadas
    """
    try:
        from dotenv import load_dotenv
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return {}
        
        if override:
            load_dotenv(file_path_obj, override=True)
        else:
            load_dotenv(file_path_obj, override=False)
        
        return dict(os.environ)
    except ImportError:
        return {}
    except Exception:
        return {}


def get_env_prefix(prefix: str, lowercase: bool = False) -> Dict[str, str]:
    """
    Obtener todas las variables con un prefijo.
    
    Args:
        prefix: Prefijo
        lowercase: Si True, convierte keys a lowercase
    
    Returns:
        Diccionario con variables
    """
    result = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            final_key = key.lower() if lowercase else key
            result[final_key] = value
    return result

