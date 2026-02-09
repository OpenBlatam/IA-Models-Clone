"""
Request logging middleware for FastAPI.
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests with timing information.
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        
        logger.info(
            f"Request started: {method} {path} "
            f"from {client_ip} "
            f"{f'?{query_params}' if query_params else ''}"
        )
        
        try:
            response = await call_next(request)
            
            process_time = time.time() - start_time
            status_code = response.status_code
            
            logger.info(
                f"Request completed: {method} {path} "
                f"status={status_code} "
                f"duration={process_time:.3f}s"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {method} {path} "
                f"error={str(e)} "
                f"duration={process_time:.3f}s",
                exc_info=True
            )
            raise






