"""
Logging Middleware - Middleware de Logging
===========================================

Middleware para logging de requests.
"""

import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging de requests"""
    
    async def dispatch(self, request: Request, call_next):
        """Procesar request y loguear"""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Procesar request
        try:
            response = await call_next(request)
            
            # Calcular duración
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} "
                f"in {duration:.3f}s for {request.method} {request.url.path}"
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Error processing {request.method} {request.url.path}: {e} "
                f"(duration: {duration:.3f}s)"
            )
            raise




