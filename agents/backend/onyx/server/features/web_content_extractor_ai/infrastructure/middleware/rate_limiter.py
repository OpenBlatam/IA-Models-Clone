"""
Rate limiting middleware
"""

import time
from collections import defaultdict
from typing import Dict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter simple basado en ventana deslizante"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        """Verificar si la solicitud está permitida"""
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
        
        # Registrar nueva solicitud
        self.requests[key].append(now)
        return True
    
    def get_remaining(self, key: str) -> int:
        """Obtener solicitudes restantes"""
        now = time.time()
        window_start = now - self.window_seconds
        
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        return max(0, self.max_requests - len(self.requests[key]))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limiting"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        # Solo aplicar rate limiting a endpoints de extracción
        if request.url.path.startswith("/api/v1/extract") and request.method == "POST":
            client_ip = request.client.host if request.client else "unknown"
            
            if not self.rate_limiter.is_allowed(client_ip):
                remaining = self.rate_limiter.get_remaining(client_ip)
                logger.warning(f"Rate limit excedido para {client_ip}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit excedido. Intenta de nuevo más tarde.",
                    headers={
                        "X-RateLimit-Limit": str(self.rate_limiter.max_requests),
                        "X-RateLimit-Remaining": str(remaining),
                        "Retry-After": str(self.rate_limiter.window_seconds)
                    }
                )
        
        response = await call_next(request)
        return response








