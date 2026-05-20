"""
Middleware - Middleware para la API
====================================

Middleware para logging, autenticación, rate limiting, etc.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging de requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(f"📥 {request.method} {request.url.path}")
        
        # Procesar request
        response = await call_next(request)
        
        # Calcular tiempo
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"📤 {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        # Agregar header de tiempo de procesamiento
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware para rate limiting"""
    
    def __init__(self, app: ASGIApp, max_requests: int = 100, window: float = 60.0):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window
        self.requests = {}  # client_ip -> [timestamps]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        
        # Verificar rate limit
        if not self._check_rate_limit(client_ip):
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            self.max_requests - len(self.requests.get(client_ip, []))
        )
        
        return response
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Verificar rate limit para IP"""
        import time
        now = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Limpiar requests antiguos
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip]
            if now - ts < self.window
        ]
        
        # Verificar límite
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Agregar request actual
        self.requests[client_ip].append(now)
        return True


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware para headers de seguridad"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Agregar headers de seguridad
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware para manejo de errores"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            from fastapi import HTTPException
            from fastapi.responses import JSONResponse
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e),
                    "path": str(request.url.path)
                }
            )


