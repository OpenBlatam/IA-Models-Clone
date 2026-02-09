"""
API Middleware - Middleware para FastAPI
========================================

Middleware para logging, performance monitoring, y request tracking.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

from ..tracing.logger import StructuredLogger, request_id_var, operation_name_var

logger = StructuredLogger(__name__, include_stack_trace=True)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging de requests con correlación."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request con logging."""
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        
        start_time = time.time()
        operation_name = f"{request.method} {request.url.path}"
        operation_name_var.set(operation_name)
        
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        try:
            response = await call_next(request)
            
            duration_ms = (time.time() - start_time) * 1000
            
            logger.log_performance(
                operation=operation_name,
                duration_ms=duration_ms,
                status_code=response.status_code
            )
            
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms
            )
            
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                exc_info=e
            )
            
            raise


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware para monitoreo de rendimiento."""
    
    def __init__(self, app: ASGIApp, slow_request_threshold_ms: float = 1000.0):
        """
        Inicializar middleware de performance.
        
        Args:
            app: Aplicación ASGI
            slow_request_threshold_ms: Umbral en ms para considerar request lento
        """
        super().__init__(app)
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self._request_times: list = []
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request con monitoreo de performance."""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            duration_ms = (time.time() - start_time) * 1000
            self._request_times.append(duration_ms)
            
            if duration_ms > self.slow_request_threshold_ms:
                logger.warning(
                    "Slow request detected",
                    method=request.method,
                    path=request.url.path,
                    duration_ms=duration_ms,
                    threshold_ms=self.slow_request_threshold_ms
                )
            
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "Request error",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                exc_info=e
            )
            raise
    
    def get_statistics(self) -> dict:
        """Obtener estadísticas de performance."""
        if not self._request_times:
            return {}
        
        import statistics
        return {
            "total_requests": len(self._request_times),
            "avg_duration_ms": statistics.mean(self._request_times),
            "min_duration_ms": min(self._request_times),
            "max_duration_ms": max(self._request_times),
            "median_duration_ms": statistics.median(self._request_times),
            "p95_duration_ms": statistics.quantiles(self._request_times, n=20)[18] if len(self._request_times) > 1 else self._request_times[0],
            "p99_duration_ms": statistics.quantiles(self._request_times, n=100)[98] if len(self._request_times) > 1 else self._request_times[0]
        }
    
    def reset_statistics(self):
        """Resetear estadísticas."""
        self._request_times.clear()

