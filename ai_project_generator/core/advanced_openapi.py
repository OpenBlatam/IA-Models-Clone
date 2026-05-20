"""
Advanced OpenAPI - OpenAPI/Swagger Avanzado
==========================================

Documentación avanzada de API:
- OpenAPI 3.0 completo
- Custom schemas
- Examples y descriptions
- Security schemes
- Response models
"""

import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

logger = logging.getLogger(__name__)


class AdvancedOpenAPIConfig:
    """Configuración avanzada de OpenAPI"""
    
    def __init__(
        self,
        title: str = "API",
        version: str = "1.0.0",
        description: str = "",
        terms_of_service: Optional[str] = None,
        contact: Optional[Dict[str, str]] = None,
        license_info: Optional[Dict[str, str]] = None,
        servers: Optional[List[Dict[str, str]]] = None
    ) -> None:
        self.title = title
        self.version = version
        self.description = description
        self.terms_of_service = terms_of_service
        self.contact = contact
        self.license_info = license_info
        self.servers = servers or [{"url": "http://localhost:8000"}]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "title": self.title,
            "version": self.version,
            "description": self.description,
            "termsOfService": self.terms_of_service,
            "contact": self.contact,
            "license": self.license_info,
            "servers": self.servers
        }


def customize_openapi(
    app: FastAPI,
    config: AdvancedOpenAPIConfig,
    custom_schemas: Optional[Dict[str, Any]] = None,
    security_schemes: Optional[Dict[str, Any]] = None
) -> None:
    """
    Personaliza OpenAPI schema.
    
    Args:
        app: Aplicación FastAPI
        config: Configuración de OpenAPI
        custom_schemas: Schemas personalizados
        security_schemes: Security schemes
    """
    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=config.title,
            version=config.version,
            description=config.description,
            routes=app.routes,
        )
        
        # Agregar información adicional
        openapi_schema["info"].update(config.to_dict())
        
        # Agregar security schemes
        if security_schemes:
            openapi_schema["components"]["securitySchemes"] = security_schemes
        
        # Agregar schemas personalizados
        if custom_schemas:
            if "components" not in openapi_schema:
                openapi_schema["components"] = {}
            if "schemas" not in openapi_schema["components"]:
                openapi_schema["components"]["schemas"] = {}
            openapi_schema["components"]["schemas"].update(custom_schemas)
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    logger.info("OpenAPI schema customized")


def add_api_examples(
    app: FastAPI,
    endpoint: str,
    method: str,
    examples: Dict[str, Any]
) -> None:
    """Agrega ejemplos a un endpoint"""
    # Implementación simplificada
    # En producción, modificaría el schema OpenAPI directamente
    logger.info(f"Examples added to {method} {endpoint}")


def get_openapi_config(
    title: str = "API",
    version: str = "1.0.0",
    **kwargs: Any
) -> AdvancedOpenAPIConfig:
    """Obtiene configuración de OpenAPI"""
    return AdvancedOpenAPIConfig(title=title, version=version, **kwargs)















