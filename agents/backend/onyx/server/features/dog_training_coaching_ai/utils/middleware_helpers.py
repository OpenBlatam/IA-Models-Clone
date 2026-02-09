"""
Middleware Helpers
==================
Utilidades para crear middlewares personalizados.
"""

from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
import time

from .logger import get_logger
from .request_id import get_request_id

logger = get_logger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware para medir tiempo de respuesta."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Medir tiempo de procesamiento."""
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        logger.info(
            "request_timing",
            path=request.url.path,
            method=request.method,
            duration=duration,
            status_code=response.status_code
        )
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging detallado de requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Loggear request detallado."""
        request_id = get_request_id(request)
        
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=request.client.host if request.client else None
        )
        
        try:
            response = await call_next(request)
            
            logger.info(
                "request_completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code
            )
            
            return response
        except Exception as e:
            logger.error(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e)
            )
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware para agregar headers de seguridad."""
    
    def __init__(self, app, additional_headers: Optional[dict] = None):
        super().__init__(app)
        self.additional_headers = additional_headers or {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Agregar headers de seguridad."""
        response = await call_next(request)
        
        # Headers de seguridad estándar
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            **self.additional_headers
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class CORSHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware para manejar CORS."""
    
    def __init__(self, app, allowed_origins: list, allowed_methods: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Manejar CORS."""
        origin = request.headers.get("origin")
        
        response = await call_next(request)
        
        if origin and origin in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response

