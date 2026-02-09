"""
Security Middleware
"""

import logging
from typing import Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware de seguridad"""
    
    def __init__(self, app, require_api_key: bool = False):
        super().__init__(app)
        self.require_api_key = require_api_key
    
    async def dispatch(self, request: Request, call_next):
        # Validar headers de seguridad
        self._validate_security_headers(request)
        
        # Validar API key si es requerido
        if self.require_api_key and not self._is_public_endpoint(request):
            if not self._validate_api_key(request):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required"
                )
        
        response = await call_next(request)
        
        # Agregar headers de seguridad
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response
    
    def _validate_security_headers(self, request: Request):
        """Valida headers de seguridad en requests"""
        # Validar tamaño de request
        if hasattr(request, "headers") and "content-length" in request.headers:
            content_length = int(request.headers.get("content-length", 0))
            max_size = 10 * 1024 * 1024  # 10MB
            if content_length > max_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request too large. Maximum size: {max_size} bytes"
                )
    
    def _validate_api_key(self, request: Request) -> bool:
        """Valida API key"""
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if not api_key:
            return False
        
        # Remover "Bearer " si está presente
        if api_key.startswith("Bearer "):
            api_key = api_key[7:]
        
        # TODO: Validar contra base de datos o configuración
        # Por ahora, aceptar cualquier key (en producción, validar)
        return len(api_key) > 0
    
    def _is_public_endpoint(self, request: Request) -> bool:
        """Verifica si el endpoint es público"""
        public_paths = ["/health", "/docs", "/redoc", "/openapi.json"]
        return request.url.path in public_paths




