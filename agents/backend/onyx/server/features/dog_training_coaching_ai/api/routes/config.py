"""
Configuration Endpoint
======================
Endpoint para información de configuración (sin datos sensibles).
"""

from fastapi import APIRouter
from typing import Dict, Any
from ...config.app_config import get_config
from ...utils.config_validator import validate_config, get_missing_config_keys

router = APIRouter(prefix="/api/v1", tags=["config"])


@router.get("/config/info")
async def get_config_info() -> Dict[str, Any]:
    """
    Obtener información de configuración (sin datos sensibles).
    
    Returns:
        Información de configuración
    """
    config = get_config()
    
    return {
        "app_name": config.app_name,
        "app_version": config.app_version,
        "host": config.host,
        "port": config.port,
        "environment": config.environment,
        "debug": config.debug,
        "cors_enabled": len(config.cors_origins) > 0,
        "openrouter_configured": bool(config.openrouter_api_key),
        "cache_enabled": True,  # Asumiendo que cache está habilitado
        "rate_limiting_enabled": True
    }


@router.get("/config/validate")
async def validate_config_endpoint() -> Dict[str, Any]:
    """
    Validar configuración de la aplicación.
    
    Returns:
        Resultado de la validación
    """
    try:
        validate_config()
        missing_keys = get_missing_config_keys()
        
        return {
            "valid": True,
            "missing_keys": missing_keys,
            "message": "Configuration is valid" if not missing_keys else "Configuration has missing optional keys"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "message": "Configuration validation failed"
        }

