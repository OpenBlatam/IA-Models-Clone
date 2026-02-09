"""
Timeout Middleware
Adds request timeout handling to prevent long-running requests
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import asyncio
import logging

logger = logging.getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeouts"""
    
    def __init__(self, app, timeout: float = 30.0):
        """
        Initialize timeout middleware
        
        Args:
            app: FastAPI application
            timeout: Request timeout in seconds
        """
        super().__init__(app)
        self.timeout = timeout
        # Paths excluded from timeout
        self.excluded_paths = {
            "/health",
            "/health/",
            "/health/live",
            "/docs",
            "/openapi.json",
            "/redoc"
        }
    
    async def dispatch(self, request: Request, call_next):
        # Skip timeout for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        try:
            # Execute with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )
            return response
            
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout: {request.method} {request.url.path} "
                f"(timeout: {self.timeout}s)"
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timeout",
                    "message": f"Request exceeded timeout of {self.timeout} seconds",
                    "path": request.url.path
                }
            )
        except Exception as e:
            logger.error(f"Timeout middleware error: {e}", exc_info=True)
            raise















