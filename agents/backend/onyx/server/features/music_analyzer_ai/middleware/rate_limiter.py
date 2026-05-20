"""
Middleware de rate limiting
"""

import time
from collections import defaultdict
from typing import Callable, Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter simple en memoria"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict = defaultdict(list)
        self.logger = logger
    
    def is_allowed(self, key: str) -> bool:
        """Verifica si una petición está permitida"""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Limpiar requests antiguos
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        # Verificar límite
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # Registrar nueva petición
        self.requests[key].append(now)
        return True
    
    def get_remaining(self, key: str) -> int:
        """Obtiene el número de peticiones restantes"""
        now = time.time()
        window_start = now - self.window_seconds
        
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        return max(0, self.max_requests - len(self.requests[key]))
    
    def reset(self, key: Optional[str] = None):
        """Resetea el contador para una clave o todas"""
        if key:
            self.requests.pop(key, None)
        else:
            self.requests.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limiting para FastAPI"""
    
    def __init__(self, app, rate_limiter: RateLimiter = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Obtener clave del cliente (IP)
        client_ip = request.client.host if request.client else "unknown"
        
        # Verificar rate limit
        if not self.rate_limiter.is_allowed(client_ip):
            remaining = self.rate_limiter.get_remaining(client_ip)
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again later.",
                headers={
                    "X-RateLimit-Limit": str(self.rate_limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + self.rate_limiter.window_seconds)
                }
            )
        
        # Continuar con la petición
        response = await call_next(request)
        
        # Agregar headers de rate limit
        remaining = self.rate_limiter.get_remaining(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.rate_limiter.window_seconds)
        
        return response

