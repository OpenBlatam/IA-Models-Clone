"""
Rate Limiting Middleware
"""

import logging
import time
from typing import Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuración de rate limiting"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10


class RateLimiter:
    """Rate limiter con ventana deslizante"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests: Dict[str, list] = defaultdict(list)
        self.lock = Lock()
    
    def _get_client_id(self, request: Request) -> str:
        """Obtiene identificador único del cliente"""
        # Prioridad: API key > IP address
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Obtener IP real (considerando proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            ip = forwarded_for.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        return f"ip:{ip}"
    
    def is_allowed(self, client_id: str) -> tuple[bool, Optional[str]]:
        """
        Verifica si el cliente puede hacer una petición
        
        Returns:
            (allowed, message)
        """
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Limpiar requests antiguos
            client_requests[:] = [req_time for req_time in client_requests if now - req_time < 86400]  # 24 horas
            
            # Verificar límites
            requests_last_minute = sum(1 for req_time in client_requests if now - req_time < 60)
            requests_last_hour = sum(1 for req_time in client_requests if now - req_time < 3600)
            requests_last_day = len(client_requests)
            
            # Verificar límites
            if requests_last_minute >= self.config.requests_per_minute:
                return False, f"Rate limit exceeded: {self.config.requests_per_minute} requests per minute"
            
            if requests_last_hour >= self.config.requests_per_hour:
                return False, f"Rate limit exceeded: {self.config.requests_per_hour} requests per hour"
            
            if requests_last_day >= self.config.requests_per_day:
                return False, f"Rate limit exceeded: {self.config.requests_per_day} requests per day"
            
            # Registrar request
            client_requests.append(now)
            
            return True, None
    
    def get_remaining(self, client_id: str) -> Dict[str, int]:
        """Obtiene requests restantes para el cliente"""
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            requests_last_minute = sum(1 for req_time in client_requests if now - req_time < 60)
            requests_last_hour = sum(1 for req_time in client_requests if now - req_time < 3600)
            requests_last_day = len(client_requests)
            
            return {
                "remaining_per_minute": max(0, self.config.requests_per_minute - requests_last_minute),
                "remaining_per_hour": max(0, self.config.requests_per_hour - requests_last_hour),
                "remaining_per_day": max(0, self.config.requests_per_day - requests_last_day),
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limiting para FastAPI"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        # Excluir health checks y docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        client_id = self.rate_limiter._get_client_id(request)
        allowed, message = self.rate_limiter.is_allowed(client_id)
        
        if not allowed:
            remaining = self.rate_limiter.get_remaining(client_id)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=message,
                headers={
                    "X-RateLimit-Limit": str(self.rate_limiter.config.requests_per_minute),
                    "X-RateLimit-Remaining": str(remaining["remaining_per_minute"]),
                    "Retry-After": "60"
                }
            )
        
        response = await call_next(request)
        
        # Agregar headers de rate limit
        remaining = self.rate_limiter.get_remaining(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.config.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining["remaining_per_minute"])
        
        return response


# Singleton global
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Obtiene instancia singleton del rate limiter"""
    global _rate_limiter
    if _rate_limiter is None:
        from ..config import get_settings
        settings = get_settings()
        config = RateLimitConfig(
            requests_per_minute=settings.rate_limit_per_minute
        )
        _rate_limiter = RateLimiter(config)
    return _rate_limiter




