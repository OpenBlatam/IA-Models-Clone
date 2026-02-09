"""
Configuration Utilities
=======================

Utilidades para manejo de configuración y variables de entorno.
"""

import os
from typing import Any, Optional, Union, Dict, List
from pathlib import Path
from .exceptions import ConfigurationError


def get_env(
    key: str,
    default: Optional[Any] = None,
    required: bool = False,
    env_type: type = str
) -> Optional[Any]:
    """
    Obtener variable de entorno con validación.
    
    Args:
        key: Nombre de la variable de entorno
        default: Valor por defecto si no existe
        required: Si es requerida (lanza excepción si no existe)
        env_type: Tipo esperado (str, int, float, bool)
    
    Returns:
        Valor de la variable de entorno
    
    Raises:
        ConfigurationError: Si es requerida y no existe
    """
    value = os.getenv(key, default)
    
    if value is None and required:
        raise ConfigurationError(
            f"Required environment variable '{key}' is not set",
            error_code="MISSING_ENV_VAR",
            details={"key": key}
        )
    
    if value is None:
        return None
    
    # Convertir tipo
    try:
        if env_type == bool:
            # Manejar strings booleanos
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)
        elif env_type == int:
            return int(value)
        elif env_type == float:
            return float(value)
        elif env_type == list:
            # Separar por comas
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
            return value
        else:
            return env_type(value)
    except (ValueError, TypeError) as e:
        raise ConfigurationError(
            f"Invalid value for environment variable '{key}': {value} (expected {env_type.__name__})",
            error_code="INVALID_ENV_VAR",
            details={"key": key, "value": value, "expected_type": env_type.__name__},
            cause=e
        )


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Obtener variable de entorno como booleano.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
    
    Returns:
        Valor booleano
    """
    return get_env(key, default=default, env_type=bool)


def get_env_int(key: str, default: Optional[int] = None, required: bool = False) -> Optional[int]:
    """
    Obtener variable de entorno como entero.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
        required: Si es requerida
    
    Returns:
        Valor entero
    """
    return get_env(key, default=default, required=required, env_type=int)


def get_env_float(key: str, default: Optional[float] = None, required: bool = False) -> Optional[float]:
    """
    Obtener variable de entorno como float.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
        required: Si es requerida
    
    Returns:
        Valor float
    """
    return get_env(key, default=default, required=required, env_type=float)


def get_env_list(key: str, default: Optional[List[str]] = None, separator: str = ",") -> Optional[List[str]]:
    """
    Obtener variable de entorno como lista.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
        separator: Separador para dividir la lista
    
    Returns:
        Lista de strings
    """
    value = get_env(key, default=default)
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(separator) if item.strip()]
    return [str(value)]


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Asegurar que un directorio existe, creándolo si es necesario.
    
    Args:
        path: Ruta del directorio
    
    Returns:
        Path del directorio
    
    Raises:
        ConfigurationError: Si no se puede crear el directorio
    """
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    except (OSError, PermissionError) as e:
        raise ConfigurationError(
            f"Cannot create directory: {path}",
            error_code="DIRECTORY_CREATION_ERROR",
            details={"path": str(path)},
            cause=e
        )


def validate_file_exists(path: Union[str, Path], required: bool = False) -> Optional[Path]:
    """
    Validar que un archivo existe.
    
    Args:
        path: Ruta del archivo
        required: Si es requerido
    
    Returns:
        Path del archivo si existe
    
    Raises:
        ConfigurationError: Si es requerido y no existe
    """
    file_path = Path(path)
    if not file_path.exists():
        if required:
            raise ConfigurationError(
                f"Required file does not exist: {path}",
                error_code="FILE_NOT_FOUND",
                details={"path": str(path)}
            )
        return None
    return file_path


def load_config_from_env(prefix: str = "ROBOT_") -> Dict[str, Any]:
    """
    Cargar todas las variables de entorno con un prefijo.
    
    Args:
        prefix: Prefijo para filtrar variables
    
    Returns:
        Diccionario con variables de entorno (sin el prefijo)
    """
    config = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            config[config_key] = value
    return config

