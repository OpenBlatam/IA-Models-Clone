"""
Debug Middleware - Middleware de debugging
==========================================

Middleware para debugging de requests y responses.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .debug_logger import get_debug_logger
from .error_tracker import get_error_tracker

logger = logging.getLogger(__name__)


class DebugMiddleware(BaseHTTPMiddleware):
    """
    Middleware de debugging que:
    - Loggea todos los requests/responses
    - Trackea errores
    - Mide tiempos de respuesta
    - Captura contexto de requests
    """
    
    def __init__(self, app: ASGIApp, enable_debug: bool = True):
        super().__init__(app)
        self.enable_debug = enable_debug
        self.debug_logger = get_debug_logger()
        self.error_tracker = get_error_tracker()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesa request con debugging"""
        if not self.enable_debug:
            return await call_next(request)
        
        # Obtener request ID
        request_id = request.headers.get("X-Request-ID", "unknown")
        
        # Establecer contexto
        self.debug_logger.set_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None
        )
        
        # Log request
        start_time = time.time()
        self.debug_logger.log_request(
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            headers=dict(request.headers)
        )
        
        try:
            # Procesar request
            response = await call_next(request)
            
            # Calcular duración
            duration = time.time() - start_time
            
            # Log response
            self.debug_logger.log_response(
                status_code=response.status_code,
                duration=duration,
                headers=dict(response.headers)
            )
            
            # Agregar headers de debug
            response.headers["X-Debug-Duration"] = str(round(duration, 4))
            response.headers["X-Debug-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Trackear error
            duration = time.time() - start_time
            self.error_tracker.track_error(
                error=e,
                context={
                    "method": request.method,
                    "path": request.url.path,
                    "duration": duration
                },
                request_id=request_id
            )
            
            # Log error
            self.debug_logger.log_exception(
                exception=e,
                context={
                    "method": request.method,
                    "path": request.url.path,
                    "duration": duration
                }
            )
            
            raise


class PerformanceDebugMiddleware(BaseHTTPMiddleware):
    """Middleware para debugging de performance"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.debug_logger = get_debug_logger()
        self.slow_requests: list = []
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Detecta requests lentos"""
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        # Alertar si es lento (> 1 segundo)
        if duration > 1.0:
            self.debug_logger.warning(
                f"Slow request: {request.method} {request.url.path} ({duration:.4f}s)",
                method=request.method,
                path=request.url.path,
                duration=duration
            )
            self.slow_requests.append({
                "method": request.method,
                "path": request.url.path,
                "duration": duration,
                "timestamp": time.time()
            })
        
        response.headers["X-Response-Time"] = str(round(duration, 4))
        
        return response















