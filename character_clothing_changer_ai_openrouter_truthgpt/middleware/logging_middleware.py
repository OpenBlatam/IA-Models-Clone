"""
Logging Middleware
==================

FastAPI middleware for request/response logging.
"""

import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Callable

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.
    
    Features:
    - Request logging with method, path, client
    - Response logging with status code, duration
    - Error logging
    - Configurable log levels
    """
    
    def __init__(
        self,
        app,
        log_requests: bool = True,
        log_responses: bool = True,
        log_body: bool = False,
        exclude_paths: list = None
    ):
        """
        Initialize logging middleware.
        
        Args:
            app: FastAPI application
            log_requests: Whether to log requests
            log_responses: Whether to log responses
            log_body: Whether to log request/response bodies
            exclude_paths: List of paths to exclude from logging
        """
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_body = log_body
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json", "/redoc"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with logging.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response with logging
        """
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            client_host = request.client.host if request.client else "unknown"
            logger.info(
                f"Request: {request.method} {request.url.path} "
                f"from {client_host}"
            )
            
            if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        logger.debug(f"Request body: {body[:500]}...")
                except Exception as e:
                    logger.warning(f"Failed to read request body: {e}")
        
        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log response
            if self.log_responses:
                status_code = response.status_code
                log_level = logging.INFO if status_code < 400 else logging.WARNING
                logger.log(
                    log_level,
                    f"Response: {request.method} {request.url.path} "
                    f"-> {status_code} ({duration:.3f}s)"
                )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Error processing {request.method} {request.url.path}: {e} "
                f"({duration:.3f}s)",
                exc_info=True
            )
            raise

