"""
Auth Middleware - Middleware de Autenticación
=============================================

Middleware para autenticación en FastAPI.
"""

import logging
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)
security = HTTPBearer()


class AuthMiddleware:
    """Middleware de autenticación"""
    
    def __init__(self, auth_service: Optional[AuthService] = None):
        """
        Inicializar middleware
        
        Args:
            auth_service: Instancia de AuthService
        """
        self.auth_service = auth_service or AuthService()
    
    async def verify_token(
        self,
        credentials: HTTPAuthorizationCredentials = security
    ) -> dict:
        """
        Verificar token JWT
        
        Args:
            credentials: Credenciales HTTP
            
        Returns:
            Payload del token
            
        Raises:
            HTTPException: Si el token es inválido
        """
        token = credentials.credentials
        payload = self.auth_service.verify_token(token)
        
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido o expirado",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
    
    def require_role(self, role: str):
        """
        Decorador para requerir rol
        
        Args:
            role: Rol requerido
            
        Returns:
            Función decoradora
        """
        async def role_checker(credentials: HTTPAuthorizationCredentials = security):
            payload = await self.verify_token(credentials)
            
            if not self.auth_service.has_role(payload, role):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Se requiere rol: {role}"
                )
            
            return payload
        
        return role_checker




