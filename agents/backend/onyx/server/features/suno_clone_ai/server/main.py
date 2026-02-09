"""
Server Main - Funciones base y entry points del módulo de servidor

Rol en el Ecosistema IA:
- API REST, endpoints, FastAPI
- Exponer funcionalidades de IA como API
- Punto de entrada principal del sistema
"""

from typing import Optional
from fastapi import FastAPI
from .service import ServerService
from .app import create_app
from .routes import setup_routes
from auth.main import get_auth_service
from access.main import get_access_service
from chat.main import get_chat_service
from tracing.main import get_tracing_service


# Instancia global del servicio
_server_service: Optional[ServerService] = None
_app: Optional[FastAPI] = None


def get_server_service() -> ServerService:
    """
    Obtiene la instancia global del servicio del servidor.
    
    Returns:
        ServerService: Servicio del servidor
    """
    global _server_service
    if _server_service is None:
        auth_service = get_auth_service()
        access_service = get_access_service()
        chat_service = get_chat_service()
        tracing_service = get_tracing_service()
        _server_service = ServerService(
            auth_service=auth_service,
            access_service=access_service,
            chat_service=chat_service,
            tracing_service=tracing_service
        )
    return _server_service


def get_app() -> FastAPI:
    """
    Obtiene la aplicación FastAPI.
    
    Returns:
        FastAPI: Aplicación FastAPI
    """
    global _app
    if _app is None:
        _app = create_app()
        server_service = get_server_service()
        setup_routes(_app, server_service)
    return _app


def initialize_server() -> ServerService:
    """
    Inicializa el servidor.
    Debe llamarse al inicio de la aplicación.
    
    Returns:
        ServerService: Servicio inicializado
    """
    return get_server_service()

