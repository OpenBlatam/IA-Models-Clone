"""
Rate Limit Middleware
=====================

Middleware para rate limiting.
"""

import logging
from typing import Callable
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limiting."""
    
    def __init__(self, app, rate_limiter=None, max_requests: int = 100, window_seconds: int = 60):
        """
        Inicializar middleware.
        
        Args:
            app: Aplicación FastAPI
            rate_limiter: Rate limiter
            max_requests: Máximo de solicitudes
            window_seconds: Ventana de tiempo
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._logger = logger
    
    def _get_client_key(self, request: Request) -> str:
        """Obtener clave del cliente."""
        # Intentar obtener de header X-User-ID
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"
        
        # Usar IP como fallback
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Procesar request con rate limiting."""
        if self.rate_limiter:
            client_key = self._get_client_key(request)
            allowed, time_remaining = self.rate_limiter.is_allowed(
                client_key,
                max_requests=self.max_requests,
                window_seconds=self.window_seconds
            )
            
            if not allowed:
                self._logger.warning(f"Rate limit exceeded for {client_key}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {int(time_remaining)} seconds",
                    headers={"Retry-After": str(int(time_remaining))}
                )
        
        response = await call_next(request)
        return response




