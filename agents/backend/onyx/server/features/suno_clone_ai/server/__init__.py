"""
Server Module - Servidor y Endpoints
Servidor FastAPI, endpoints, rutas, y API REST.

Rol en el Ecosistema IA:
- API REST, endpoints, FastAPI
- Exponer funcionalidades de IA como API
- Punto de entrada principal del sistema

Reglas de Importación:
- Puede importar: Todos los módulos anteriores
- Es el punto de entrada del sistema
- NO debe ser importado por otros módulos (excepto main.py)
"""

from .base import BaseServer
from .service import ServerService
from .app import create_app
from .routes import setup_routes
from .main import (
    get_server_service,
    get_app,
    initialize_server,
)

__all__ = [
    # Clases principales
    "BaseServer",
    "ServerService",
    # Funciones principales
    "create_app",
    "setup_routes",
    # Funciones de acceso rápido
    "get_server_service",
    "get_app",
    "initialize_server",
]

