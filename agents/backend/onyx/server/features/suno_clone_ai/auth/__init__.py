"""
Auth Module - Autenticación y Autorización
Maneja autenticación de usuarios, tokens JWT, OAuth, y sesiones.

Rol en el Ecosistema IA:
- Autenticación de usuarios, tokens JWT
- Control de acceso a modelos, rate limiting por usuario
- Seguridad del sistema de IA

Reglas de Importación:
- Puede importar: db, redis, configs
- NO debe importar: módulos de negocio (chat, agents, etc.)
- Usa inyección de dependencias
"""

from .base import BaseAuthenticator
from .service import AuthService
from .jwt_handler import JWTHandler
from .main import (
    get_auth_service,
    authenticate,
    verify_token,
    create_token,
    initialize_auth,
)

__all__ = [
    # Clases principales
    "BaseAuthenticator",
    "AuthService",
    "JWTHandler",
    # Funciones de acceso rápido
    "get_auth_service",
    "authenticate",
    "verify_token",
    "create_token",
    "initialize_auth",
]

