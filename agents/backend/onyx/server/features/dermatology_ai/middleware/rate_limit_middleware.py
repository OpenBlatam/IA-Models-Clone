"""
Middleware de rate limiting
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ..utils.rate_limiter import RateLimiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware para rate limiting"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        """
        Inicializa el middleware
        
        Args:
            app: Aplicación FastAPI
            rate_limiter: Instancia de RateLimiter
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        """Procesa el request con rate limiting"""
        # Obtener identificador (IP o user_id si está autenticado)
        identifier = request.client.host if request.client else "unknown"
        
        # Verificar rate limit
        allowed, info = self.rate_limiter.is_allowed(identifier)
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Demasiadas requests. Intenta de nuevo en {info.get('retry_after', 60)} segundos.",
                    "retry_after": info.get("retry_after", 60),
                    "reset_at": info.get("reset_at")
                },
                headers={
                    "X-RateLimit-Limit": str(info.get("limit", 100)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": info.get("reset_at", ""),
                    "Retry-After": str(info.get("retry_after", 60))
                }
            )
        
        # Agregar headers de rate limit
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(info.get("limit", 100))
        response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = info.get("reset_at", "")
        
        return response






