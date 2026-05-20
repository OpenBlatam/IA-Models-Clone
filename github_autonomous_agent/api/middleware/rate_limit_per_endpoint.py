"""
Rate Limiting por Endpoint - Rate limiting granular por ruta.
"""

import time
from typing import Dict, Optional, Callable
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from collections import defaultdict, deque

from config.logging_config import get_logger
from config.di_setup import get_service

logger = get_logger(__name__)


class EndpointRateLimit:
    """Configuración de rate limit para un endpoint."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        key_func: Optional[Callable[[Request], str]] = None
    ):
        """
        Inicializar rate limit de endpoint.
        
        Args:
            requests_per_minute: Requests por minuto
            requests_per_hour: Requests por hora
            requests_per_day: Requests por día
            key_func: Función para generar key de rate limit (opcional)
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.key_func = key_func or (lambda req: req.client.host if req.client else "unknown")


class EndpointRateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware para rate limiting por endpoint."""
    
    def __init__(self, app, endpoint_limits: Optional[Dict[str, EndpointRateLimit]] = None):
        """
        Inicializar middleware.
        
        Args:
            app: Aplicación FastAPI
            endpoint_limits: Diccionario de límites por endpoint
        """
        super().__init__(app)
        self.endpoint_limits = endpoint_limits or {}
        self.request_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: {
            "minute": deque(),
            "hour": deque(),
            "day": deque()
        })
        self.cleanup_interval = 300  # Limpiar cada 5 minutos
        self.last_cleanup = time.time()
    
    def _get_endpoint_key(self, request: Request) -> str:
        """
        Obtener key del endpoint.
        
        Args:
            request: Request de FastAPI
            
        Returns:
            Key del endpoint
        """
        return f"{request.method}:{request.url.path}"
    
    def _get_client_key(self, request: Request, endpoint_limit: EndpointRateLimit) -> str:
        """
        Obtener key del cliente.
        
        Args:
            request: Request de FastAPI
            endpoint_limit: Configuración de rate limit
            
        Returns:
            Key del cliente
        """
        return endpoint_limit.key_func(request)
    
    def _check_rate_limit(
        self,
        endpoint_key: str,
        client_key: str,
        endpoint_limit: EndpointRateLimit
    ) -> tuple[bool, Optional[Dict[str, int]]]:
        """
        Verificar rate limit.
        
        Args:
            endpoint_key: Key del endpoint
            client_key: Key del cliente
            endpoint_limit: Configuración de rate limit
            
        Returns:
            Tupla (permitido, remaining)
        """
        full_key = f"{endpoint_key}:{client_key}"
        history = self.request_history[full_key]
        now = time.time()
        
        # Limpiar historial antiguo
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400
        
        # Limpiar requests antiguos
        while history["minute"] and history["minute"][0] < minute_ago:
            history["minute"].popleft()
        while history["hour"] and history["hour"][0] < hour_ago:
            history["hour"].popleft()
        while history["day"] and history["day"][0] < day_ago:
            history["day"].popleft()
        
        # Verificar límites
        minute_count = len(history["minute"])
        hour_count = len(history["hour"])
        day_count = len(history["day"])
        
        if minute_count >= endpoint_limit.requests_per_minute:
            return False, {
                "minute": 0,
                "hour": max(0, endpoint_limit.requests_per_hour - hour_count),
                "day": max(0, endpoint_limit.requests_per_day - day_count)
            }
        
        if hour_count >= endpoint_limit.requests_per_hour:
            return False, {
                "minute": max(0, endpoint_limit.requests_per_minute - minute_count),
                "hour": 0,
                "day": max(0, endpoint_limit.requests_per_day - day_count)
            }
        
        if day_count >= endpoint_limit.requests_per_day:
            return False, {
                "minute": max(0, endpoint_limit.requests_per_minute - minute_count),
                "hour": max(0, endpoint_limit.requests_per_hour - hour_count),
                "day": 0
            }
        
        # Registrar request
        history["minute"].append(now)
        history["hour"].append(now)
        history["day"].append(now)
        
        return True, {
            "minute": endpoint_limit.requests_per_minute - minute_count - 1,
            "hour": endpoint_limit.requests_per_hour - hour_count - 1,
            "day": endpoint_limit.requests_per_day - day_count - 1
        }
    
    def _cleanup_old_entries(self):
        """Limpiar entradas antiguas."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        keys_to_remove = []
        for key, history in self.request_history.items():
            # Si todas las historias están vacías, eliminar
            if not any(history.values()):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.request_history[key]
        
        self.last_cleanup = now
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Procesar request con rate limiting por endpoint.
        
        Args:
            request: Request de FastAPI
            call_next: Función para continuar
            
        Returns:
            Response
        """
        endpoint_key = self._get_endpoint_key(request)
        
        # Verificar si hay rate limit para este endpoint
        endpoint_limit = self.endpoint_limits.get(endpoint_key)
        if not endpoint_limit:
            # No hay rate limit para este endpoint
            return await call_next(request)
        
        # Limpiar entradas antiguas periódicamente
        self._cleanup_old_entries()
        
        # Obtener key del cliente
        client_key = self._get_client_key(request, endpoint_limit)
        
        # Verificar rate limit
        allowed, remaining = self._check_rate_limit(endpoint_key, client_key, endpoint_limit)
        
        if not allowed:
            logger.warning(
                f"Rate limit excedido para {endpoint_key} - cliente: {client_key}"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "detail": f"Rate limit exceeded for {endpoint_key}",
                    "retry_after": 60,
                    "limits": {
                        "per_minute": endpoint_limit.requests_per_minute,
                        "per_hour": endpoint_limit.requests_per_hour,
                        "per_day": endpoint_limit.requests_per_day
                    }
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit-Minute": str(endpoint_limit.requests_per_minute),
                    "X-RateLimit-Limit-Hour": str(endpoint_limit.requests_per_hour),
                    "X-RateLimit-Limit-Day": str(endpoint_limit.requests_per_day),
                    "X-RateLimit-Remaining-Minute": "0",
                    "X-RateLimit-Remaining-Hour": str(remaining.get("hour", 0)),
                    "X-RateLimit-Remaining-Day": str(remaining.get("day", 0))
                }
            )
        
        # Agregar headers de rate limit
        response = await call_next(request)
        if remaining:
            response.headers["X-RateLimit-Limit-Minute"] = str(endpoint_limit.requests_per_minute)
            response.headers["X-RateLimit-Limit-Hour"] = str(endpoint_limit.requests_per_hour)
            response.headers["X-RateLimit-Limit-Day"] = str(endpoint_limit.requests_per_day)
            response.headers["X-RateLimit-Remaining-Minute"] = str(remaining.get("minute", 0))
            response.headers["X-RateLimit-Remaining-Hour"] = str(remaining.get("hour", 0))
            response.headers["X-RateLimit-Remaining-Day"] = str(remaining.get("day", 0))
        
        return response


def get_default_endpoint_limits() -> Dict[str, EndpointRateLimit]:
    """
    Obtener límites por defecto para endpoints.
    
    Returns:
        Diccionario de límites por endpoint
    """
    return {
        "POST:/api/v1/tasks": EndpointRateLimit(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000
        ),
        "POST:/api/v1/batch/tasks": EndpointRateLimit(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000
        ),
        "POST:/api/v1/llm/generate": EndpointRateLimit(
            requests_per_minute=20,
            requests_per_hour=300,
            requests_per_day=3000
        ),
        "GET:/api/v1/tasks": EndpointRateLimit(
            requests_per_minute=100,
            requests_per_hour=2000,
            requests_per_day=20000
        ),
        "POST:/api/v1/agent/start": EndpointRateLimit(
            requests_per_minute=5,
            requests_per_hour=50,
            requests_per_day=500
        )
    }



