"""
Middleware para monitoreo de rendimiento
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..utils.performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware para monitorear rendimiento de requests"""
    
    async def dispatch(self, request: Request, call_next):
        """Procesa request y mide tiempo"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Registrar request
            is_error = response.status_code >= 400
            get_performance_monitor().record_request(duration, is_error=is_error)
            
            # Agregar header con tiempo de respuesta
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            get_performance_monitor().record_request(duration, is_error=True)
            raise




