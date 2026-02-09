"""
Configs Module - Configuraciones del Sistema
Centraliza todas las configuraciones, variables de entorno, y settings.

Rol en el Ecosistema IA:
- Base del sistema, sin dependencias
- Centraliza configuraciones de modelos, APIs, hiperparámetros
- Punto único de acceso a settings

Reglas de Importación:
- Este módulo NO debe importar otros módulos del proyecto
- Puede importar solo librerías externas (pydantic, etc.)
"""

from .base import BaseConfig
from .service import ConfigService
from .settings import Settings
from .main import (
    get_settings,
    get_config_service,
    get_setting,
    reload_config,
    initialize_config,
)

__all__ = [
    # Clases principales
    "BaseConfig",
    "ConfigService",
    "Settings",
    # Funciones de acceso rápido
    "get_settings",
    "get_config_service",
    "get_setting",
    "reload_config",
    "initialize_config",
]

