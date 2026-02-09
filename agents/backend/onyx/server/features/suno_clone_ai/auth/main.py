"""
Auth Main - Funciones base y entry points del módulo de autenticación

Rol en el Ecosistema IA:
- Autenticación de usuarios, tokens JWT
- Control de acceso a modelos, rate limiting por usuario
- Seguridad del sistema de IA
"""

from typing import Optional, Dict, Any
from .service import AuthService
from .jwt_handler import JWTHandler
from db.main import get_db_service
from redis.main import get_redis_service


# Instancia global del servicio
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """
    Obtiene la instancia global del servicio de autenticación.
    
    Returns:
        AuthService: Servicio de autenticación
    """
    global _auth_service
    if _auth_service is None:
        db_service = get_db_service()
        redis_service = get_redis_service()
        jwt_handler = JWTHandler()
        # TODO: Crear authenticator concreto
        from .base import BaseAuthenticator
        authenticator = None  # Debe ser implementado
        _auth_service = AuthService(
            authenticator=authenticator,
            jwt_handler=jwt_handler,
            db_service=db_service,
            redis_service=redis_service
        )
    return _auth_service


async def authenticate(credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Autentica un usuario.
    
    Args:
        credentials: Credenciales del usuario
        
    Returns:
        Token y datos del usuario o None
    """
    service = get_auth_service()
    return await service.authenticate(credentials)


async def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verifica un token JWT.
    
    Args:
        token: Token JWT
        
    Returns:
        Payload del token o None
    """
    service = get_auth_service()
    return await service.verify_token(token)


def create_token(payload: Dict[str, Any], expires_in: int = 3600) -> str:
    """
    Crea un token JWT.
    
    Args:
        payload: Datos a incluir en el token
        expires_in: Tiempo de expiración en segundos
        
    Returns:
        Token JWT
    """
    jwt_handler = JWTHandler()
    return jwt_handler.encode(payload, expires_in)


def initialize_auth() -> AuthService:
    """
    Inicializa el sistema de autenticación.
    
    Returns:
        AuthService: Servicio inicializado
    """
    return get_auth_service()

