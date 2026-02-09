"""
Configuration Validator
=======================
Utilidades para validar configuración.
"""

from typing import Optional, List, Dict, Any
from ..config.app_config import get_config
from ..core.exceptions import ValidationException
from ..core.error_codes import ErrorCode


def validate_config() -> bool:
    """
    Validar configuración de la aplicación.
    
    Returns:
        True si la configuración es válida
        
    Raises:
        ValidationException: Si la configuración es inválida
    """
    config = get_config()
    errors = []
    
    # Validar OpenRouter API key
    if not config.openrouter_api_key:
        errors.append("OPENROUTER_API_KEY is required")
    
    # Validar puerto
    if not (1 <= config.port <= 65535):
        errors.append(f"Port must be between 1 and 65535, got {config.port}")
    
    # Validar host
    if not config.host:
        errors.append("Host cannot be empty")
    
    # Validar CORS origins
    if not config.cors_origins:
        errors.append("CORS origins cannot be empty")
    
    if errors:
        raise ValidationException(
            f"Configuration validation failed: {', '.join(errors)}",
            error_code=ErrorCode.CONFIGURATION_ERROR
        )
    
    return True


def get_missing_config_keys() -> List[str]:
    """
    Obtener claves de configuración faltantes.
    
    Returns:
        Lista de claves faltantes
    """
    config = get_config()
    missing = []
    
    if not config.openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")
    
    return missing


def validate_environment() -> Dict[str, Any]:
    """
    Validar entorno de ejecución.
    
    Returns:
        Diccionario con información del entorno
    """
    import sys
    import platform
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_valid": True
    }

