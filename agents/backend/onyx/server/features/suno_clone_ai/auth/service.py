"""
Auth Service - Servicio de autenticación
"""

from typing import Optional, Dict, Any
from .base import BaseAuthenticator
from .jwt_handler import JWTHandler
from db.service import DatabaseService
from redis.service import RedisService


class AuthService:
    """Servicio para gestionar autenticación"""

    def __init__(
        self,
        authenticator: BaseAuthenticator,
        jwt_handler: Optional[JWTHandler] = None,
        db_service: Optional[DatabaseService] = None,
        redis_service: Optional[RedisService] = None
    ):
        """Inicializa el servicio de autenticación"""
        self.authenticator = authenticator
        self.jwt_handler = jwt_handler or JWTHandler()
        self.db_service = db_service
        self.redis_service = redis_service

    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Autentica un usuario"""
        return await self.authenticator.authenticate(credentials)

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica un token"""
        return await self.authenticator.verify_token(token)

