"""
Configuración desde variables de entorno mejorada
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env si existe
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """Obtiene variable de entorno"""
    value = os.getenv(key, default)
    
    if required and not value:
        raise ValueError(f"Variable de entorno requerida no encontrada: {key}")
    
    return value


def get_env_int(key: str, default: int = 0) -> int:
    """Obtiene variable de entorno como entero"""
    value = get_env(key, str(default))
    try:
        return int(value)
    except ValueError:
        return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """Obtiene variable de entorno como booleano"""
    value = get_env(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_list(key: str, default: Optional[list] = None, separator: str = ",") -> list:
    """Obtiene variable de entorno como lista"""
    value = get_env(key)
    if not value:
        return default or []
    return [item.strip() for item in value.split(separator) if item.strip()]




