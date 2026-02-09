"""
Base middleware class for consistent middleware implementation.

Provides a foundation for all middleware with common patterns.
"""

import logging
import time
from typing import Callable, Optional, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class BaseMiddleware(BaseHTTPMiddleware):
    """Base middleware class with common functionality."""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = kwargs
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Dispatch request through middleware.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
        
        Returns:
            Response
        """
        start_time = time.time()
        
        # Process request
        request = await self.process_request(request)
        
        try:
            # Call next middleware/handler
            response = await call_next(request)
            
            # Process response
            response = await self.process_response(request, response)
            
            # Record metrics
            elapsed = time.time() - start_time
            await self.record_metrics(request, response, elapsed)
            
            return response
            
        except Exception as e:
            # Handle errors
            return await self.process_error(request, e)
    
    async def process_request(self, request: Request) -> Request:
        """
        Process incoming request.
        
        Override in subclasses to add request processing.
        
        Args:
            request: Incoming request
        
        Returns:
            Processed request
        """
        return request
    
    async def process_response(
        self,
        request: Request,
        response: Response
    ) -> Response:
        """
        Process outgoing response.
        
        Override in subclasses to add response processing.
        
        Args:
            request: Original request
            response: Response to process
        
        Returns:
            Processed response
        """
        return response
    
    async def process_error(
        self,
        request: Request,
        error: Exception
    ) -> Response:
        """
        Process errors.
        
        Override in subclasses to add error handling.
        
        Args:
            request: Original request
            error: Exception that occurred
        
        Returns:
            Error response
        """
        self.logger.error(f"Error in {request.url.path}: {error}", exc_info=True)
        raise error
    
    async def record_metrics(
        self,
        request: Request,
        response: Response,
        elapsed: float
    ) -> None:
        """
        Record metrics for the request.
        
        Override in subclasses to add custom metrics.
        
        Args:
            request: Original request
            response: Response
            elapsed: Elapsed time in seconds
        """
        # Default: just log
        self.logger.debug(
            f"{request.method} {request.url.path} - {response.status_code} - {elapsed:.3f}s"
        )

