"""
OpenAPI Helpers - Utilidades para mejorar documentación OpenAPI
===============================================================

Funciones helper para mejorar y personalizar la documentación OpenAPI
generada automáticamente por FastAPI.
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

logger = logging.getLogger(__name__)


def customize_openapi_schema(
    app: FastAPI,
    title: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None,
    contact: Optional[Dict[str, str]] = None,
    license_info: Optional[Dict[str, str]] = None,
    servers: Optional[List[Dict[str, str]]] = None,
    tags_metadata: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Personalizar el schema OpenAPI de la aplicación.
    
    Args:
        app: Aplicación FastAPI
        title: Título de la API (opcional)
        version: Versión de la API (opcional)
        description: Descripción de la API (opcional)
        contact: Información de contacto (opcional)
        license_info: Información de licencia (opcional)
        servers: Lista de servidores (opcional)
        tags_metadata: Metadata de tags (opcional)
    """
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=title or app.title,
            version=version or app.version,
            description=description or app.description,
            routes=app.routes,
        )
        
        # Agregar información de contacto
        if contact:
            openapi_schema["info"]["contact"] = contact
        
        # Agregar información de licencia
        if license_info:
            openapi_schema["info"]["license"] = license_info
        
        # Agregar servidores
        if servers:
            openapi_schema["servers"] = servers
        
        # Agregar metadata de tags
        if tags_metadata:
            openapi_schema["tags"] = tags_metadata
        
        # Agregar componentes de seguridad
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token authentication. Format: Bearer <token>"
            }
        }
        
        # Agregar seguridad global
        openapi_schema["security"] = [{"BearerAuth": []}]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi


def add_example_to_schema(
    schema: Dict[str, Any],
    example: Any,
    examples: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Agregar ejemplo a un schema OpenAPI.
    
    Args:
        schema: Schema OpenAPI
        example: Ejemplo a agregar
        examples: Lista de ejemplos adicionales (opcional)
        
    Returns:
        Schema con ejemplo agregado
    """
    if example is not None:
        schema["example"] = example
    
    if examples:
        schema["examples"] = {f"example_{i}": ex for i, ex in enumerate(examples)}
    
    return schema


def create_tag_metadata(
    name: str,
    description: str,
    external_docs: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Crear metadata de tag para OpenAPI.
    
    Args:
        name: Nombre del tag
        description: Descripción del tag
        external_docs: Documentación externa (opcional)
        
    Returns:
        Diccionario con metadata del tag
    """
    tag = {
        "name": name,
        "description": description
    }
    
    if external_docs:
        tag["externalDocs"] = external_docs
    
    return tag

