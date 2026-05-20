"""
Middleware personalizado para la API.
"""

import time
from typing import Callable, Optional, Any, Tuple
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para logging detallado de requests.
    
    Registra información sobre cada request incluyendo:
    - Método HTTP y path
    - IP del cliente
    - Tiempo de procesamiento
    - Código de estado
    - User-Agent (opcional)
    """
    
    SKIP_PATHS = ["/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"]
    
    def _get_client_info(self, request: Request) -> dict:
        """
        Obtener información del cliente.
        
        Args:
            request: Request de FastAPI
            
        Returns:
            Diccionario con información del cliente
        """
        client_ip = request.client.host if request.client else "unknown"
        
        # Intentar obtener IP real (detrás de proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return {
            "ip": client_ip,
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "referer": request.headers.get("Referer")
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Procesar request y agregar logging detallado.
        
        Args:
            request: Request de FastAPI
            call_next: Función para continuar con el siguiente middleware
            
        Returns:
            Response de FastAPI
        """
        # Skip logging para paths específicos
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)
        
        start_time = time.time()
        client_info = self._get_client_info(request)
        
        # Log de request entrante
        logger.info(
            f"→ {request.method} {request.url.path} "
            f"[{client_info['ip']}] "
            f"UA: {client_info['user_agent'][:50]}"
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Determinar nivel de log basado en código de estado
            if response.status_code < 400:
                log_level = logger.info
            elif response.status_code < 500:
                log_level = logger.warning
            else:
                log_level = logger.error
            
            # Log de response
            log_level(
                f"← {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Time: {process_time:.3f}s "
                f"[{client_info['ip']}]"
            )
            
            # Agregar headers de métricas
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            response.headers["X-Request-ID"] = getattr(request.state, "request_id", "unknown")
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"✗ {request.method} {request.url.path} "
                f"Error: {type(e).__name__} "
                f"Time: {process_time:.3f}s "
                f"[{client_info['ip']}]",
                exc_info=True
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para manejo centralizado de errores.
    
    Captura excepciones no manejadas y retorna respuestas JSON consistentes.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Procesar request con manejo de errores robusto.
        
        Args:
            request: Request de FastAPI
            call_next: Función para continuar con el siguiente middleware
            
        Returns:
            Response de FastAPI o JSONResponse con error
        """
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            # HTTPException ya tiene status_code y detail
            logger.warning(
                f"HTTPException en {request.method} {request.url.path}: "
                f"{e.status_code} - {e.detail}"
            )
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": True,
                    "detail": e.detail,
                    "path": request.url.path,
                    "method": request.method
                }
            )
        except ValueError as e:
            # Errores de validación
            logger.warning(
                f"Error de validación en {request.method} {request.url.path}: {e}"
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "detail": f"Error de validación: {str(e)}",
                    "path": request.url.path,
                    "method": request.method
                }
            )
        except Exception as e:
            # Errores inesperados
            error_id = getattr(request.state, "request_id", "unknown")
            logger.error(
                f"Error no manejado en {request.method} {request.url.path} "
                f"[{error_id}]: {type(e).__name__}: {e}",
                exc_info=True
            )
            
            # Solo mostrar mensaje detallado en modo debug
            show_details = settings.DEBUG
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "detail": "Error interno del servidor",
                    "error_id": error_id,
                    "message": str(e) if show_details else None,
                    "type": type(e).__name__ if show_details else None
                }
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware para agregar headers de seguridad."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Agregar headers de seguridad a la respuesta."""
        response = await call_next(request)
        
        # Headers de seguridad
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # HSTS (solo en producción)
        if not settings.DEBUG:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware para rate limiting global con fallback en memoria.
    
    Soporta servicio de rate limiting externo o fallback en memoria.
    """
    
    def __init__(self, app, rate_limit_service: Optional[Any] = None):
        """
        Inicializar middleware de rate limiting.
        
        Args:
            app: Aplicación FastAPI
            rate_limit_service: Servicio de rate limiting (opcional)
        """
        super().__init__(app)
        self.rate_limit_service = rate_limit_service
        self.rate_limits: dict = {}  # Fallback en memoria si no hay servicio
        self.default_limit = getattr(settings, "RATE_LIMIT_PER_MINUTE", 100)
        self.window_seconds = 60
        self.cleanup_interval = 300  # Limpiar cada 5 minutos
        self.last_cleanup = time.time()
        
        if not self.rate_limit_service:
            logger.info(f"Rate limiting usando fallback en memoria: {self.default_limit} req/min")
    
    def _get_client_id(self, request: Request) -> str:
        """
        Obtener identificador único del cliente.
        
        Args:
            request: Request de FastAPI
            
        Returns:
            ID único del cliente (IP)
        """
        # Intentar obtener IP real (detrás de proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _cleanup_old_entries(self):
        """Limpiar entradas antiguas del rate limiting en memoria."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff = now - self.window_seconds
        clients_to_remove = []
        
        for client_id, requests in self.rate_limits.items():
            self.rate_limits[client_id] = [
                req_time for req_time in requests
                if req_time > cutoff
            ]
            if not self.rate_limits[client_id]:
                clients_to_remove.append(client_id)
        
        for client_id in clients_to_remove:
            del self.rate_limits[client_id]
        
        self.last_cleanup = now
        logger.debug(f"Rate limit cleanup: {len(self.rate_limits)} clientes activos")
    
    def _check_rate_limit(self, client_id: str) -> Tuple[bool, Optional[int]]:
        """
        Verificar rate limit para un cliente.
        
        Args:
            client_id: ID único del cliente
            
        Returns:
            Tupla con (permitido, requests_restantes)
        """
        # Intentar usar servicio externo
        if self.rate_limit_service:
            try:
                remaining = self.rate_limit_service.check_rate_limit(client_id, cost=1)
                return True, remaining
            except Exception as e:
                logger.warning(f"Error en rate limit service, usando fallback: {e}")
                # Continuar con fallback
        
        # Fallback: rate limiting en memoria
        self._cleanup_old_entries()
        
        now = time.time()
        
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []
        
        # Limpiar requests antiguos para este cliente
        cutoff = now - self.window_seconds
        self.rate_limits[client_id] = [
            req_time for req_time in self.rate_limits[client_id]
            if req_time > cutoff
        ]
        
        # Verificar límite
        current_requests = len(self.rate_limits[client_id])
        if current_requests >= self.default_limit:
            remaining = 0
            return False, remaining
        
        # Registrar request
        self.rate_limits[client_id].append(now)
        remaining = self.default_limit - current_requests - 1
        return True, remaining
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Procesar request con rate limiting.
        
        Args:
            request: Request de FastAPI
            call_next: Función para continuar con el siguiente middleware
            
        Returns:
            Response de FastAPI o JSONResponse con error 429
        """
        # Skip rate limiting para health checks y docs
        skip_paths = ["/health", "/docs", "/openapi.json", "/redoc", "/", "/favicon.ico"]
        if request.url.path in skip_paths:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        allowed, remaining = self._check_rate_limit(client_id)
        
        if not allowed:
            logger.warning(
                f"Rate limit excedido para {client_id} en {request.method} {request.url.path}"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "detail": "Demasiados requests. Por favor intenta más tarde.",
                    "retry_after": self.window_seconds,
                    "limit": self.default_limit,
                    "window_seconds": self.window_seconds
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.default_limit),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Agregar headers de rate limit
        response = await call_next(request)
        if remaining is not None:
            response.headers["X-RateLimit-Limit"] = str(self.default_limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response

