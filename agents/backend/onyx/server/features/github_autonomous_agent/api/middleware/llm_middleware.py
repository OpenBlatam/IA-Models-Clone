"""
Middleware para rutas LLM.

Incluye:
- Rate limiting automático
- Logging de requests
- Validación de headers
- Métricas de performance
"""

import time
from typing import Callable
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config.logging_config import get_logger
from core.services.llm import get_advanced_rate_limiter, RateLimitStrategy
from core.constants import ErrorMessages

logger = get_logger(__name__)


class LLMRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware de rate limiting para rutas LLM.
    
    Aplica rate limiting automático basado en:
    - IP del cliente
    - Ruta del endpoint
    - Modelo solicitado (si está disponible)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        default_limit: int = 100,
        window_seconds: int = 60,
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    ):
        """
        Inicializar middleware.
        
        Args:
            app: Aplicación ASGI
            default_limit: Límite por defecto de requests
            window_seconds: Ventana de tiempo en segundos
            strategy: Estrategia de rate limiting
        """
        super().__init__(app)
        self.rate_limiter = get_advanced_rate_limiter()
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.strategy = strategy
        
        # Configurar rate limits por ruta
        self._configure_route_limits()
    
    def _configure_route_limits(self):
        """Configurar límites específicos por ruta."""
        # Límites más estrictos para endpoints costosos
        route_limits = {
            "/api/v1/llm/generate": {"limit": 50, "window": 60},
            "/api/v1/llm/generate-parallel": {"limit": 20, "window": 60},
            "/api/v1/llm/generate-stream": {"limit": 30, "window": 60},
            "/api/v1/llm/ab-test/create": {"limit": 10, "window": 300},
            "/api/v1/llm/tests/create-suite": {"limit": 10, "window": 300},
        }
        
        for route, config in route_limits.items():
            self.rate_limiter.configure(
                key=f"route:{route}",
                limit=config["limit"],
                window_seconds=config["window"],
                strategy=self.strategy
            )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Procesar request con rate limiting.
        
        Args:
            request: Request HTTP
            call_next: Siguiente middleware/handler
            
        Returns:
            Response HTTP
        """
        # Solo aplicar a rutas LLM
        if not request.url.path.startswith("/api/v1/llm"):
            return await call_next(request)
        
        # Obtener clave de rate limiting
        client_ip = request.client.host if request.client else "unknown"
        route_key = f"route:{request.url.path}"
        ip_key = f"ip:{client_ip}"
        
        # Verificar rate limit por ruta
        route_limit = self.rate_limiter.is_allowed(route_key, tokens=1)
        if not route_limit.allowed:
            logger.warning(
                f"Rate limit excedido para ruta {request.url.path} desde IP {client_ip}"
            )
            return Response(
                content=ErrorMessages.LLM_RATE_LIMIT_EXCEEDED,
                status_code=429,
                headers={
                    "Retry-After": str(int(route_limit.retry_after)),
                    "X-RateLimit-Limit": str(route_limit.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(route_limit.reset_at))
                }
            )
        
        # Verificar rate limit por IP (más permisivo)
        ip_limit = self.rate_limiter.is_allowed(ip_key, tokens=1)
        if not ip_limit.allowed:
            logger.warning(f"Rate limit excedido para IP {client_ip}")
            return Response(
                content=ErrorMessages.LLM_RATE_LIMIT_EXCEEDED,
                status_code=429,
                headers={
                    "Retry-After": str(int(ip_limit.retry_after)),
                    "X-RateLimit-Limit": str(ip_limit.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(ip_limit.reset_at))
                }
            )
        
        # Agregar headers de rate limit
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(route_limit.limit)
        response.headers["X-RateLimit-Remaining"] = str(route_limit.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(route_limit.reset_at))
        
        return response


class LLMLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware de logging para rutas LLM.
    
    Registra:
    - Requests y responses
    - Latencia
    - Errores
    - Métricas de uso
    """
    
    def __init__(self, app: ASGIApp, log_requests: bool = True, log_responses: bool = False):
        """
        Inicializar middleware.
        
        Args:
            app: Aplicación ASGI
            log_requests: Si registrar requests
            log_responses: Si registrar responses (puede ser verbose)
        """
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Procesar request con logging.
        
        Args:
            request: Request HTTP
            call_next: Siguiente middleware/handler
            
        Returns:
            Response HTTP
        """
        # Solo aplicar a rutas LLM
        if not request.url.path.startswith("/api/v1/llm"):
            return await call_next(request)
        
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request
        if self.log_requests:
            logger.info(
                f"LLM Request: {request.method} {request.url.path} "
                f"from {client_ip} "
                f"query_params={dict(request.query_params)}"
            )
        
        try:
            response = await call_next(request)
            latency_ms = (time.time() - start_time) * 1000
            
            # Log response
            logger.info(
                f"LLM Response: {request.method} {request.url.path} "
                f"status={response.status_code} "
                f"latency={latency_ms:.1f}ms"
            )
            
            # Agregar header de latencia
            response.headers["X-Response-Time"] = f"{latency_ms:.1f}ms"
            
            return response
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                f"LLM Error: {request.method} {request.url.path} "
                f"error={str(e)} "
                f"latency={latency_ms:.1f}ms",
                exc_info=True
            )
            raise


class LLMValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware de validación para rutas LLM.
    
    Valida:
    - Headers requeridos
    - Content-Type
    - Tamaño de request body
    """
    
    def __init__(
        self,
        app: ASGIApp,
        max_body_size: int = 10 * 1024 * 1024  # 10MB
    ):
        """
        Inicializar middleware.
        
        Args:
            app: Aplicación ASGI
            max_body_size: Tamaño máximo del body en bytes
        """
        super().__init__(app)
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Procesar request con validación.
        
        Args:
            request: Request HTTP
            call_next: Siguiente middleware/handler
            
        Returns:
            Response HTTP
        """
        # Solo aplicar a rutas LLM
        if not request.url.path.startswith("/api/v1/llm"):
            return await call_next(request)
        
        # Validar Content-Type para POST/PUT
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                logger.warning(
                    f"Content-Type inválido: {content_type} "
                    f"para {request.method} {request.url.path}"
                )
                return Response(
                    content="Content-Type debe ser application/json",
                    status_code=400
                )
        
        # Validar tamaño del body
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_body_size:
                    logger.warning(
                        f"Request body demasiado grande: {size} bytes "
                        f"para {request.url.path}"
                    )
                    return Response(
                        content=f"Request body excede el límite de {self.max_body_size} bytes",
                        status_code=413
                    )
            except ValueError:
                pass
        
        return await call_next(request)



