"""
Middleware de rate limiting para la API (optimizado)

Usa helpers optimizados para mejor rendimiento y consistencia.
"""

import logging
from typing import Callable
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

try:
    from config.settings import settings
except ImportError:
    from ..config.settings import settings

from ..api.utils.rate_limit_helpers import check_rate_limit, get_rate_limit_info
from ..api.utils.request_helpers import get_client_ip

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Middleware para limitar la tasa de peticiones (optimizado).
    
    Usa helpers optimizados para mejor rendimiento y consistencia.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = None,
        window_seconds: int = None
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or settings.rate_limit_requests
        self.window_seconds = window_seconds or settings.rate_limit_window
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Procesa el request con rate limiting.
        
        Args:
            request: Request object
            call_next: Función para continuar el pipeline
            
        Returns:
            Response con headers de rate limit
        """
        # Obtener identificador del cliente
        client_id = self._get_client_id(request)
        
        # Verificar rate limit usando helper optimizado
        is_allowed, retry_after = check_rate_limit(
            identifier=client_id,
            max_requests=self.requests_per_minute,
            window_seconds=self.window_seconds
        )
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per {self.window_seconds} seconds.",
                headers={"Retry-After": str(retry_after) if retry_after else str(self.window_seconds)}
            )
        
        # Procesar request
        response = await call_next(request)
        
        # Agregar headers de rate limit usando helper
        rate_limit_info = get_rate_limit_info(
            identifier=client_id,
            max_requests=self.requests_per_minute,
            window_seconds=self.window_seconds
        )
        
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_limit_info["reset_in"])
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """
        Obtiene un identificador único del cliente.
        
        Prioriza user_id sobre IP para mejor tracking.
        """
        # Intentar obtener user_id del header o query
        user_id = request.headers.get("X-User-ID") or request.query_params.get("user_id")
        if user_id:
            return f"user:{user_id}"
        
        # Usar IP como fallback usando helper
        client_ip = get_client_ip(request)
        return f"ip:{client_ip}"

