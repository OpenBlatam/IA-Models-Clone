"""
Logging Middleware
==================
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging estructurado."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request con logging."""
        start_time = time.time()
        
        # Log request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            path=request.url.path,
            method=request.method,
            client_ip=request.client.host if request.client else None
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "request_completed",
                status_code=response.status_code,
                process_time=round(process_time, 3)
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "request_failed",
                error=str(e),
                process_time=round(process_time, 3)
            )
            raise

