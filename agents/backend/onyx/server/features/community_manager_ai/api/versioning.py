"""
API Versioning - Versionado de API
===================================

Sistema de versionado de API.
"""

from fastapi import APIRouter
from typing import Dict, Any

# Versión actual de la API
CURRENT_VERSION = "v1"
SUPPORTED_VERSIONS = ["v1"]


def create_versioned_router(version: str = CURRENT_VERSION) -> APIRouter:
    """
    Crear router versionado
    
    Args:
        version: Versión de la API
        
    Returns:
        Router versionado
    """
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Versión {version} no soportada")
    
    prefix = f"/api/{version}"
    return APIRouter(prefix=prefix)


def get_api_info() -> Dict[str, Any]:
    """
    Obtener información de la API
    
    Returns:
        Dict con información de versiones
    """
    return {
        "current_version": CURRENT_VERSION,
        "supported_versions": SUPPORTED_VERSIONS,
        "deprecated_versions": [],
        "endpoints": {
            "posts": f"/api/{CURRENT_VERSION}/posts",
            "memes": f"/api/{CURRENT_VERSION}/memes",
            "calendar": f"/api/{CURRENT_VERSION}/calendar",
            "platforms": f"/api/{CURRENT_VERSION}/platforms",
            "analytics": f"/api/{CURRENT_VERSION}/analytics",
            "templates": f"/api/{CURRENT_VERSION}/templates",
            "export": f"/api/{CURRENT_VERSION}/export",
            "webhooks": f"/api/{CURRENT_VERSION}/webhooks",
            "dashboard": f"/api/{CURRENT_VERSION}/dashboard",
            "batch": f"/api/{CURRENT_VERSION}/batch",
            "backup": f"/api/{CURRENT_VERSION}/backup",
            "monitoring": f"/api/{CURRENT_VERSION}/monitoring"
        }
    }




