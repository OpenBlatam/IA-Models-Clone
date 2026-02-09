"""
Configuration Helpers - Utilidades para configuración
=====================================================

Funciones helper para trabajar con configuración del servidor MCP.
"""

import logging
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Obtener valor booleano desde variable de entorno.
    
    Args:
        key: Nombre de la variable de entorno
        default: Valor por defecto
        
    Returns:
        Valor booleano
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int) -> int:
    """
    Obtener valor entero desde variable de entorno.
    
    Args:
        key: Nombre de la variable de entorno
        default: Valor por defecto
        
    Returns:
        Valor entero
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer value for {key}: {value}, using default: {default}")
        return default


def get_env_list(key: str, default: Optional[List[str]] = None, separator: str = ",") -> List[str]:
    """
    Obtener lista desde variable de entorno.
    
    Args:
        key: Nombre de la variable de entorno
        default: Valor por defecto
        separator: Separador para dividir la lista
        
    Returns:
        Lista de strings
    """
    value = os.getenv(key)
    if value is None:
        return default or []
    
    return [item.strip() for item in value.split(separator) if item.strip()]


def validate_config_path(path: Optional[str], must_exist: bool = False) -> Optional[str]:
    """
    Validar y normalizar ruta de configuración.
    
    Args:
        path: Ruta a validar
        must_exist: Si la ruta debe existir
        
    Returns:
        Ruta normalizada o None
        
    Raises:
        ValueError: Si la ruta es inválida
    """
    if path is None:
        return None
    
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    return str(path_obj.absolute())


def get_config_summary(config: Dict[str, Any], sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Obtener resumen de configuración ocultando valores sensibles.
    
    Args:
        config: Diccionario de configuración
        sensitive_keys: Lista de keys a ocultar (default: ["secret_key", "password"])
        
    Returns:
        Resumen de configuración con valores sensibles ocultos
    """
    if sensitive_keys is None:
        sensitive_keys = ["secret_key", "password", "token", "api_key"]
    
    def mask_sensitive(data: Any, path: str = "") -> Any:
        """Ocultar valores sensibles recursivamente"""
        if isinstance(data, dict):
            return {
                key: mask_sensitive(value, f"{path}.{key}" if path else key)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [mask_sensitive(item, path) for item in data]
        else:
            # Verificar si la key es sensible
            if any(sensitive in path.lower() for sensitive in sensitive_keys):
                return "***HIDDEN***"
            return data
    
    return mask_sensitive(config)


def load_config_file(file_path: str) -> Dict[str, Any]:
    """
    Cargar archivo de configuración (JSON o YAML).
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Diccionario con configuración
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el formato no es soportado
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    import json
    import yaml
    
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in [".yaml", ".yml"]:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML file: {e}") from e
        elif path.suffix.lower() == ".json":
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {e}") from e
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

