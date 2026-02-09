"""
Monitoring Middleware - Middleware de Monitoreo
================================================

Middleware para monitoreo de requests.
"""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..services.monitoring_service import MonitoringService


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware para monitoreo de requests"""
    
    def __init__(self, app, monitoring_service: MonitoringService):
        """
        Inicializar middleware
        
        Args:
            app: Aplicación FastAPI
            monitoring_service: Instancia de MonitoringService
        """
        super().__init__(app)
        self.monitoring = monitoring_service
    
    async def dispatch(self, request: Request, call_next):
        """Procesar request y monitorear"""
        start_time = time.time()
        
        # Incrementar contador de requests
        self.monitoring.increment_counter("http_requests_total")
        self.monitoring.increment_counter(f"http_requests_{request.method.lower()}")
        
        # Procesar request
        try:
            response = await call_next(request)
            
            # Registrar duración
            duration = time.time() - start_time
            self.monitoring.record_timing("http_request_duration", duration)
            
            # Contador por status code
            self.monitoring.increment_counter(f"http_status_{response.status_code}")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitoring.record_timing("http_request_duration", duration)
            self.monitoring.increment_counter("http_errors_total")
            raise




