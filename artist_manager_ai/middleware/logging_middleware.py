"""
Logging Middleware
==================

Middleware para logging estructurado.
"""

import logging
import time
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware de logging."""
    
    def __init__(self, app):
        """Inicializar middleware."""
        super().__init__(app)
        self._logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Procesar request con logging."""
        start_time = time.time()
        
        # Log request
        self._logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None
            }
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            self._logger.info(
                f"Response: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
        
        except Exception as e:
            process_time = time.time() - start_time
            self._logger.error(
                f"Error: {request.method} {request.url.path} - {str(e)}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "process_time": process_time
                },
                exc_info=True
            )
            raise




