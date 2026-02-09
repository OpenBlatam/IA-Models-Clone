"""
Authentication Middleware
"""

import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware para autenticación"""
    
    def __init__(self, app, auth_service=None):
        super().__init__(app)
        self.auth_service = auth_service
    
    async def dispatch(self, request: Request, call_next):
        # Rutas públicas que no requieren autenticación
        public_paths = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/register",
            "/api/v1/auth/login",
        ]
        
        # Verificar si la ruta es pública
        if any(request.url.path.startswith(path) for path in public_paths):
            return await call_next(request)
        
        # Verificar token de sesión
        session_id = request.headers.get("X-Session-ID") or request.cookies.get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Verificar sesión (si auth_service está disponible)
        if self.auth_service:
            user = self.auth_service.verify_session(session_id)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid or expired session")
            
            # Agregar usuario al request state
            request.state.user = user
        
        response = await call_next(request)
        return response




