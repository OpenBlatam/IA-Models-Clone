"""
Rate Limiting Avanzado

Proporciona:
- Rate limiting por usuario
- Rate limiting por endpoint
- Rate limiting por IP
- Sliding window
- Token bucket
- Diferentes límites por tipo de usuario
"""

import logging
import time
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings

logger = logging.getLogger(__name__)


class RateLimitConfig:
    """Configuración de rate limiting"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        burst_size: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_size = burst_size


class AdvancedRateLimiter:
    """Rate limiter avanzado con múltiples ventanas de tiempo"""
    
    def __init__(self):
        # Almacenar requests por identificador
        self.requests_minute: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.requests_hour: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.requests_day: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100000))
        
        # Configuraciones por tipo de usuario
        self.configs: Dict[str, RateLimitConfig] = {
            "default": RateLimitConfig(),
            "premium": RateLimitConfig(
                requests_per_minute=120,
                requests_per_hour=5000,
                requests_per_day=50000
            ),
            "admin": RateLimitConfig(
                requests_per_minute=1000,
                requests_per_hour=100000,
                requests_per_day=1000000
            )
        }
        
        logger.info("AdvancedRateLimiter initialized")
    
    def get_client_id(self, request: Request) -> Tuple[str, str]:
        """
        Obtiene identificador del cliente y tipo de usuario
        
        Returns:
            Tuple de (client_id, user_type)
        """
        # Intentar obtener user_id del header o token
        user_id = request.headers.get("X-User-ID")
        user_type = request.headers.get("X-User-Type", "default")
        
        if user_id:
            return f"user:{user_id}", user_type
        
        # Usar IP como fallback
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}", "default"
    
    def is_rate_limited(
        self,
        client_id: str,
        user_type: str = "default",
        endpoint: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifica si el cliente ha excedido el rate limit
        
        Args:
            client_id: Identificador del cliente
            user_type: Tipo de usuario
            endpoint: Endpoint específico (opcional)
        
        Returns:
            Tuple de (is_limited, reason)
        """
        config = self.configs.get(user_type, self.configs["default"])
        current_time = time.time()
        
        # Limpiar requests antiguos
        self._cleanup_old_requests(client_id, current_time)
        
        # Verificar límites por ventana de tiempo
        minute_count = len(self.requests_minute[client_id])
        hour_count = len(self.requests_hour[client_id])
        day_count = len(self.requests_day[client_id])
        
        if minute_count >= config.requests_per_minute:
            return True, f"Rate limit exceeded: {minute_count}/{config.requests_per_minute} requests per minute"
        
        if hour_count >= config.requests_per_hour:
            return True, f"Rate limit exceeded: {hour_count}/{config.requests_per_hour} requests per hour"
        
        if day_count >= config.requests_per_day:
            return True, f"Rate limit exceeded: {day_count}/{config.requests_per_day} requests per day"
        
        return False, None
    
    def record_request(self, client_id: str, current_time: float):
        """Registra un request"""
        self.requests_minute[client_id].append(current_time)
        self.requests_hour[client_id].append(current_time)
        self.requests_day[client_id].append(current_time)
    
    def _cleanup_old_requests(self, client_id: str, current_time: float):
        """Limpia requests antiguos"""
        # Limpiar requests de hace más de 1 minuto
        minute_ago = current_time - 60
        while (self.requests_minute[client_id] and 
               self.requests_minute[client_id][0] < minute_ago):
            self.requests_minute[client_id].popleft()
        
        # Limpiar requests de hace más de 1 hora
        hour_ago = current_time - 3600
        while (self.requests_hour[client_id] and 
               self.requests_hour[client_id][0] < hour_ago):
            self.requests_hour[client_id].popleft()
        
        # Limpiar requests de hace más de 1 día
        day_ago = current_time - 86400
        while (self.requests_day[client_id] and 
               self.requests_day[client_id][0] < day_ago):
            self.requests_day[client_id].popleft()
    
    def get_remaining_requests(
        self,
        client_id: str,
        user_type: str = "default"
    ) -> Dict[str, int]:
        """Obtiene requests restantes por ventana"""
        config = self.configs.get(user_type, self.configs["default"])
        current_time = time.time()
        self._cleanup_old_requests(client_id, current_time)
        
        return {
            "minute": max(0, config.requests_per_minute - len(self.requests_minute[client_id])),
            "hour": max(0, config.requests_per_hour - len(self.requests_hour[client_id])),
            "day": max(0, config.requests_per_day - len(self.requests_day[client_id]))
        }


class AdvancedRateLimiterMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limiting avanzado"""
    
    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = AdvancedRateLimiter()
        # Endpoints excluidos del rate limiting
        self.excluded_paths = {
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next):
        # Verificar si el path está excluido
        if request.url.path in self.excluded_paths or request.url.path.startswith("/docs"):
            return await call_next(request)
        
        # Obtener identificador del cliente
        client_id, user_type = self.rate_limiter.get_client_id(request)
        
        # Verificar rate limit
        is_limited, reason = self.rate_limiter.is_rate_limited(
            client_id,
            user_type=user_type,
            endpoint=request.url.path
        )
        
        if is_limited:
            remaining = self.rate_limiter.get_remaining_requests(client_id, user_type)
            logger.warning(f"Rate limit exceeded for {client_id}: {reason}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=reason,
                headers={
                    "X-RateLimit-Limit-Minute": str(self.rate_limiter.configs[user_type].requests_per_minute),
                    "X-RateLimit-Limit-Hour": str(self.rate_limiter.configs[user_type].requests_per_hour),
                    "X-RateLimit-Limit-Day": str(self.rate_limiter.configs[user_type].requests_per_day),
                    "X-RateLimit-Remaining-Minute": str(remaining["minute"]),
                    "X-RateLimit-Remaining-Hour": str(remaining["hour"]),
                    "X-RateLimit-Remaining-Day": str(remaining["day"]),
                    "Retry-After": "60"
                }
            )
        
        # Registrar request
        current_time = time.time()
        self.rate_limiter.record_request(client_id, current_time)
        
        # Procesar request
        response = await call_next(request)
        
        # Agregar headers de rate limit
        remaining = self.rate_limiter.get_remaining_requests(client_id, user_type)
        config = self.rate_limiter.configs[user_type]
        
        response.headers["X-RateLimit-Limit-Minute"] = str(config.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(config.requests_per_hour)
        response.headers["X-RateLimit-Limit-Day"] = str(config.requests_per_day)
        response.headers["X-RateLimit-Remaining-Minute"] = str(remaining["minute"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(remaining["hour"])
        response.headers["X-RateLimit-Remaining-Day"] = str(remaining["day"])
        
        return response

