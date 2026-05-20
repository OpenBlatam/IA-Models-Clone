"""
Authentication Middleware
========================

Middleware para autenticación en FastAPI.
"""

import logging
from typing import Callable
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware de autenticación."""
    
    def __init__(self, app, auth_service=None, public_paths: list = None):
        """
        Inicializar middleware.
        
        Args:
            app: Aplicación FastAPI
            auth_service: Servicio de autenticación
            public_paths: Rutas públicas (sin autenticación)
        """
        super().__init__(app)
        self.auth_service = auth_service
        self.public_paths = public_paths or ["/health", "/docs", "/openapi.json"]
        self._logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Procesar request."""
        # Verificar si es ruta pública
        if any(request.url.path.startswith(path) for path in self.public_paths):
            return await call_next(request)
        
        # Verificar autenticación
        if self.auth_service:
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing authentication token"
                )
            
            user = self.auth_service.get_user_from_token(token)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            
            # Agregar usuario al request
            request.state.user = user
        
        response = await call_next(request)
        return response




