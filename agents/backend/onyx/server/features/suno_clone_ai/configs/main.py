"""
Configs Main - Funciones base y entry points del módulo de configuraciones

Rol en el Ecosistema IA:
- Base del sistema, sin dependencias
- Centraliza todas las configuraciones (modelos, APIs, hiperparámetros)
- Punto único de acceso a settings del sistema
"""

from typing import Optional, Any, Dict
from .settings import Settings
from .service import ConfigService


# Instancia global del servicio (singleton pattern)
_config_service: Optional[ConfigService] = None


def get_settings() -> Settings:
    """
    Obtiene la instancia global de Settings.
    
    Returns:
        Settings: Instancia de configuración del sistema
    """
    return Settings()


def get_config_service() -> ConfigService:
    """
    Obtiene la instancia global del servicio de configuraciones.
    Usa patrón singleton para evitar múltiples instancias.
    
    Returns:
        ConfigService: Servicio de configuraciones
    """
    global _config_service
    if _config_service is None:
        _config_service = ConfigService()
    return _config_service


def get_setting(key: str, default: Any = None) -> Any:
    """
    Obtiene un setting específico del sistema.
    
    Args:
        key: Nombre del setting
        default: Valor por defecto si no existe
        
    Returns:
        Valor del setting o default
    """
    settings = get_settings()
    return getattr(settings, key, default)


def reload_config() -> None:
    """
    Recarga las configuraciones del sistema.
    Útil cuando se cambian variables de entorno en runtime.
    """
    global _config_service
    if _config_service:
        _config_service.reload()
    else:
        _config_service = ConfigService()


def get_all_settings() -> Dict[str, Any]:
    """
    Obtiene todos los settings como diccionario.
    
    Returns:
        Dict con todos los settings
    """
    settings = get_settings()
    return settings.dict() if hasattr(settings, 'dict') else settings.__dict__


# Entry point para inicialización
def initialize_config() -> ConfigService:
    """
    Inicializa el sistema de configuraciones.
    Debe llamarse al inicio de la aplicación.
    
    Returns:
        ConfigService: Servicio inicializado
    """
    return get_config_service()

