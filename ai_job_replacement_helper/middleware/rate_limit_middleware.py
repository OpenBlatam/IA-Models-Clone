"""
Rate Limiting Middleware
"""

import logging
import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware para rate limiting"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        # Obtener identificador del cliente
        client_id = request.client.host if request.client else "unknown"
        
        # Obtener timestamp actual
        now = time.time()
        
        # Limpiar requests antiguos (más de 1 minuto)
        self.requests[client_id] = [
            ts for ts in self.requests[client_id]
            if now - ts < 60
        ]
        
        # Verificar rate limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_id}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
        
        # Registrar request
        self.requests[client_id].append(now)
        
        # Agregar headers de rate limit
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.requests[client_id])
        )
        
        return response




