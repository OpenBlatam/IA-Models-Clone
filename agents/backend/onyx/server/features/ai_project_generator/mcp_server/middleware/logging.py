"""
Request Logging Middleware
==========================

Middleware para logging estructurado de requests y responses.
"""

import logging
import time
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para logging estructurado de requests y responses.
    
    Registra información detallada sobre cada request incluyendo:
    - Método HTTP
    - Path
    - Status code
    - Duración
    - Request ID (si está disponible)
    """
    
    def __init__(self, app, log_level: str = "info"):
        """
        Inicializar middleware de logging.
        
        Args:
            app: FastAPI application
            log_level: Nivel de logging ("debug", "info", "warning", "error")
        """
        super().__init__(app)
        self.log_level = log_level
    
    async def dispatch(self, request: Request, call_next):
        """
        Procesar request y registrar información.
        
        Args:
            request: Request object
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        start_time = time.time()
        method = request.method
        path = request.url.path
        request_id = getattr(request.state, "request_id", None)
        
        # Log request start (solo en debug)
        if self.log_level == "debug":
            logger.debug(
                f"Request started: {method} {path}",
                extra={
                    "method": method,
                    "path": path,
                    "request_id": request_id,
                    "type": "request_start"
                }
            )
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            duration = time.time() - start_time
            logger.error(
                f"Request error: {method} {path} - {e}",
                exc_info=True,
                extra={
                    "method": method,
                    "path": path,
                    "request_id": request_id,
                    "error": str(e),
                    "duration_ms": duration * 1000,
                    "type": "request_error"
                }
            )
            raise
        finally:
            duration = time.time() - start_time
            log_data = {
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration * 1000,
                "request_id": request_id,
                "type": "request_complete"
            }
            
            # Log según nivel y status code
            if status_code >= 500:
                logger.error(
                    f"Request failed: {method} {path} - {status_code}",
                    extra=log_data
                )
            elif status_code >= 400:
                logger.warning(
                    f"Request error: {method} {path} - {status_code}",
                    extra=log_data
                )
            elif self.log_level in ["info", "debug"]:
                logger.info(
                    f"Request: {method} {path} - {status_code}",
                    extra=log_data
                )
        
        return response

