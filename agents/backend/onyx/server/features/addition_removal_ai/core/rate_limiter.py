"""
Rate Limiter - Sistema de limitación de tasa de solicitudes
"""

import logging
import time
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Limitador de tasa de solicitudes"""

    def __init__(
        self,
        max_requests: int = 100,
        time_window: int = 60,
        per_user: bool = True
    ):
        """
        Inicializar el limitador.

        Args:
            max_requests: Número máximo de solicitudes
            time_window: Ventana de tiempo en segundos
            per_user: Si la limitación es por usuario
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.per_user = per_user
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())

    def is_allowed(self, identifier: str = "default") -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verificar si una solicitud está permitida.

        Args:
            identifier: Identificador (usuario, IP, etc.)

        Returns:
            Tupla (permitido, información)
        """
        now = time.time()
        key = identifier if self.per_user else "global"
        
        # Limpiar solicitudes antiguas
        requests_queue = self.requests[key]
        while requests_queue and requests_queue[0] < now - self.time_window:
            requests_queue.popleft()
        
        # Verificar límite
        if len(requests_queue) >= self.max_requests:
            oldest_request = requests_queue[0] if requests_queue else now
            reset_time = oldest_request + self.time_window
            return False, {
                "limit_exceeded": True,
                "reset_at": datetime.fromtimestamp(reset_time).isoformat(),
                "remaining": 0,
                "limit": self.max_requests
            }
        
        # Registrar solicitud
        requests_queue.append(now)
        
        return True, {
            "limit_exceeded": False,
            "remaining": self.max_requests - len(requests_queue),
            "limit": self.max_requests,
            "reset_at": datetime.fromtimestamp(now + self.time_window).isoformat()
        }

    def get_stats(self, identifier: str = "default") -> Dict[str, Any]:
        """
        Obtener estadísticas de rate limiting.

        Args:
            identifier: Identificador

        Returns:
            Estadísticas
        """
        key = identifier if self.per_user else "global"
        requests_queue = self.requests[key]
        now = time.time()
        
        # Limpiar solicitudes antiguas
        while requests_queue and requests_queue[0] < now - self.time_window:
            requests_queue.popleft()
        
        return {
            "current_requests": len(requests_queue),
            "max_requests": self.max_requests,
            "remaining": self.max_requests - len(requests_queue),
            "time_window": self.time_window,
            "reset_in": max(0, self.time_window - (now - requests_queue[0])) if requests_queue else 0
        }

    def reset(self, identifier: Optional[str] = None):
        """
        Resetear contador para un identificador.

        Args:
            identifier: Identificador (None para todos)
        """
        if identifier:
            if identifier in self.requests:
                self.requests[identifier].clear()
        else:
            self.requests.clear()


class RateLimitMiddleware:
    """Middleware para rate limiting"""

    def __init__(self, rate_limiter: RateLimiter):
        """
        Inicializar middleware.

        Args:
            rate_limiter: Instancia del limitador
        """
        self.rate_limiter = rate_limiter

    async def __call__(self, request, call_next):
        """
        Procesar solicitud con rate limiting.

        Args:
            request: Solicitud
            call_next: Función siguiente

        Returns:
            Respuesta
        """
        from fastapi import Request, HTTPException
        
        # Obtener identificador (IP o usuario)
        identifier = request.client.host if hasattr(request, 'client') else "unknown"
        
        # Verificar rate limit
        allowed, info = self.rate_limiter.is_allowed(identifier)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Try again after {info['reset_at']}",
                    "reset_at": info["reset_at"]
                }
            )
        
        # Agregar headers de rate limit
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = info["reset_at"]
        
        return response






