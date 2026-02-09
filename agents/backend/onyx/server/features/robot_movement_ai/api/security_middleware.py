"""
Security Middleware - Middleware de seguridad
==============================================

Middleware para rate limiting, autenticación y seguridad.
"""

import time
from typing import Callable, Optional, Dict, Any
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from ..tracing.logger import StructuredLogger

logger = StructuredLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware para rate limiting simple."""
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        """
        Inicializar rate limiter.
        
        Args:
            app: Aplicación ASGI
            requests_per_minute: Requests permitidos por minuto
            requests_per_hour: Requests permitidos por hora
            burst_size: Tamaño de burst permitido
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self._minute_requests: Dict[str, list] = defaultdict(list)
        self._hour_requests: Dict[str, list] = defaultdict(list)
        self._last_cleanup = time.time()
    
    def _get_client_id(self, request: Request) -> str:
        """Obtener identificador único del cliente."""
        client_host = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return client_host
    
    def _cleanup_old_requests(self):
        """Limpiar requests antiguos."""
        current_time = time.time()
        if current_time - self._last_cleanup < 60:
            return
        
        cutoff_minute = current_time - 60
        cutoff_hour = current_time - 3600
        
        for client_id in list(self._minute_requests.keys()):
            self._minute_requests[client_id] = [
                t for t in self._minute_requests[client_id] if t > cutoff_minute
            ]
            if not self._minute_requests[client_id]:
                del self._minute_requests[client_id]
        
        for client_id in list(self._hour_requests.keys()):
            self._hour_requests[client_id] = [
                t for t in self._hour_requests[client_id] if t > cutoff_hour
            ]
            if not self._hour_requests[client_id]:
                del self._hour_requests[client_id]
        
        self._last_cleanup = current_time
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request con rate limiting."""
        self._cleanup_old_requests()
        
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        minute_requests = self._minute_requests[client_id]
        hour_requests = self._hour_requests[client_id]
        
        minute_requests = [t for t in minute_requests if t > current_time - 60]
        hour_requests = [t for t in hour_requests if t > current_time - 3600]
        
        if len(minute_requests) >= self.requests_per_minute:
            logger.warning(
                "Rate limit exceeded (per minute)",
                client_id=client_id,
                requests=len(minute_requests),
                limit=self.requests_per_minute
            )
            return Response(
                content='{"error": "Rate limit exceeded. Too many requests per minute."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + 60)),
                    "Retry-After": "60"
                },
                media_type="application/json"
            )
        
        if len(hour_requests) >= self.requests_per_hour:
            logger.warning(
                "Rate limit exceeded (per hour)",
                client_id=client_id,
                requests=len(hour_requests),
                limit=self.requests_per_hour
            )
            return Response(
                content='{"error": "Rate limit exceeded. Too many requests per hour."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_hour),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + 3600)),
                    "Retry-After": "3600"
                },
                media_type="application/json"
            )
        
        minute_requests.append(current_time)
        hour_requests.append(current_time)
        
        self._minute_requests[client_id] = minute_requests
        self._hour_requests[client_id] = hour_requests
        
        response = await call_next(request)
        
        remaining_minute = max(0, self.requests_per_minute - len(minute_requests))
        remaining_hour = max(0, self.requests_per_hour - len(hour_requests))
        
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(remaining_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"] = str(remaining_hour)
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware para agregar headers de seguridad."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request agregando headers de seguridad."""
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware para limitar tamaño de requests."""
    
    def __init__(self, app: ASGIApp, max_request_size: int = 10 * 1024 * 1024):
        """
        Inicializar middleware de límite de tamaño.
        
        Args:
            app: Aplicación ASGI
            max_request_size: Tamaño máximo en bytes (default: 10MB)
        """
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request validando tamaño."""
        content_length = request.headers.get("content-length")
        
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    logger.warning(
                        "Request too large",
                        size=size,
                        max_size=self.max_request_size,
                        path=request.url.path
                    )
                    return Response(
                        content='{"error": "Request body too large"}',
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        media_type="application/json"
                    )
            except ValueError:
                pass
        
        response = await call_next(request)
        return response

