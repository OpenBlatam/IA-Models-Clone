"""
Load Balancer Middleware
Health checks y readiness para load balancers
"""

import logging
import asyncio
from typing import Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoadBalancerMiddleware(BaseHTTPMiddleware):
    """
    Middleware para integración con load balancers
    Maneja health checks, readiness, y graceful shutdown
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.health_check_path = kwargs.get('health_check_path', '/health')
        self.readiness_path = kwargs.get('readiness_path', '/ready')
        self.liveness_path = kwargs.get('liveness_path', '/live')
        self.shutdown_timeout = kwargs.get('shutdown_timeout', 30)
        self._shutting_down = False
        self._ready = True
    
    async def dispatch(self, request: Request, call_next):
        """Procesa request con load balancer logic"""
        # Health check rápido
        if request.url.path == self.health_check_path:
            return Response(
                content='{"status": "healthy"}',
                media_type="application/json",
                status_code=200
            )
        
        # Readiness check
        if request.url.path == self.readiness_path:
            if self._ready and not self._shutting_down:
                return Response(
                    content='{"status": "ready"}',
                    media_type="application/json",
                    status_code=200
                )
            else:
                return Response(
                    content='{"status": "not_ready"}',
                    media_type="application/json",
                    status_code=503
                )
        
        # Liveness check
        if request.url.path == self.liveness_path:
            return Response(
                content='{"status": "alive"}',
                media_type="application/json",
                status_code=200
            )
        
        # Rechazar requests durante shutdown
        if self._shutting_down:
            return Response(
                content='{"error": "Service is shutting down"}',
                media_type="application/json",
                status_code=503,
                headers={"Retry-After": str(self.shutdown_timeout)}
            )
        
        # Procesar request normal
        response = await call_next(request)
        
        # Agregar headers para load balancer
        response.headers["X-Service-Status"] = "ready" if self._ready else "not_ready"
        
        return response
    
    def set_ready(self, ready: bool):
        """Marca el servicio como ready/not ready"""
        self._ready = ready
        logger.info(f"Service readiness set to: {ready}")
    
    def start_shutdown(self):
        """Inicia el proceso de shutdown"""
        self._shutting_down = True
        self._ready = False
        logger.info("Service shutdown initiated")















