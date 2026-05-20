"""
Middleware Base for Color Grading AI
=====================================

Base classes for middleware with common functionality.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from functools import wraps

logger = logging.getLogger(__name__)


class BaseMiddleware(ABC):
    """
    Base class for middleware.
    
    Provides:
    - Request/response interception
    - Error handling
    - Logging
    - Metrics
    """
    
    def __init__(self, name: str):
        """
        Initialize middleware.
        
        Args:
            name: Middleware name
        """
        self.name = name
        self._enabled = True
    
    def enable(self):
        """Enable middleware."""
        self._enabled = True
        logger.info(f"Enabled middleware: {self.name}")
    
    def disable(self):
        """Disable middleware."""
        self._enabled = False
        logger.info(f"Disabled middleware: {self.name}")
    
    def is_enabled(self) -> bool:
        """Check if middleware is enabled."""
        return self._enabled
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming request.
        
        Args:
            request: Request dictionary
            
        Returns:
            Processed request
        """
        pass
    
    @abstractmethod
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process outgoing response.
        
        Args:
            response: Response dictionary
            
        Returns:
            Processed response
        """
        pass
    
    async def process_error(self, error: Exception, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process error.
        
        Args:
            error: Exception
            request: Request dictionary
            
        Returns:
            Error response
        """
        logger.error(f"Error in {self.name}: {error}")
        return {
            "error": str(error),
            "error_type": type(error).__name__,
        }
    
    def __call__(self, handler: Callable) -> Callable:
        """Make middleware callable."""
        @wraps(handler)
        async def wrapper(request: Dict[str, Any]) -> Dict[str, Any]:
            if not self._enabled:
                return await handler(request)
            
            try:
                # Process request
                processed_request = await self.process_request(request)
                
                # Call handler
                response = await handler(processed_request)
                
                # Process response
                processed_response = await self.process_response(response)
                
                return processed_response
            
            except Exception as e:
                return await self.process_error(e, request)
        
        return wrapper


class TimingMiddleware(BaseMiddleware):
    """Middleware for request timing."""
    
    def __init__(self):
        super().__init__("timing")
        self._timings: Dict[str, float] = {}
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Record start time."""
        request["_start_time"] = time.time()
        return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate duration."""
        start_time = response.get("_start_time", time.time())
        duration = time.time() - start_time
        
        response["duration"] = duration
        response["_timings"] = self._timings.copy()
        
        return response


class LoggingMiddleware(BaseMiddleware):
    """Middleware for request logging."""
    
    def __init__(self, log_level: str = "info"):
        super().__init__("logging")
        self.log_level = log_level
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Log request."""
        log_func = getattr(logger, self.log_level, logger.info)
        log_func(f"Request: {request.get('path', 'unknown')} - {request.get('method', 'unknown')}")
        return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Log response."""
        logger.info(f"Response: {response.get('status_code', 'unknown')}")
        return response




