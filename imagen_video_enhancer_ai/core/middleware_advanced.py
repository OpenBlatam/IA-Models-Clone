"""
Advanced Middleware System
===========================

Advanced middleware system for FastAPI with request/response processing.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class MiddlewareType(Enum):
    """Middleware types."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    FINALLY = "finally"


@dataclass
class MiddlewareConfig:
    """Middleware configuration."""
    enabled: bool = True
    priority: int = 0
    skip_paths: List[str] = field(default_factory=list)
    only_paths: List[str] = field(default_factory=list)


class AdvancedMiddleware(BaseHTTPMiddleware):
    """Advanced middleware with request/response processing."""
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[MiddlewareConfig] = None
    ):
        """
        Initialize advanced middleware.
        
        Args:
            app: ASGI application
            config: Middleware configuration
        """
        super().__init__(app)
        self.config = config or MiddlewareConfig()
        self.request_handlers: List[Callable] = []
        self.response_handlers: List[Callable] = []
        self.error_handlers: List[Callable] = []
        self.finally_handlers: List[Callable] = []
    
    def add_request_handler(self, handler: Callable):
        """
        Add request handler.
        
        Args:
            handler: Request handler function
        """
        self.request_handlers.append(handler)
    
    def add_response_handler(self, handler: Callable):
        """
        Add response handler.
        
        Args:
            handler: Response handler function
        """
        self.response_handlers.append(handler)
    
    def add_error_handler(self, handler: Callable):
        """
        Add error handler.
        
        Args:
            handler: Error handler function
        """
        self.error_handlers.append(handler)
    
    def add_finally_handler(self, handler: Callable):
        """
        Add finally handler.
        
        Args:
            handler: Finally handler function
        """
        self.finally_handlers.append(handler)
    
    def _should_process(self, path: str) -> bool:
        """
        Check if path should be processed.
        
        Args:
            path: Request path
            
        Returns:
            True if should process
        """
        # Check skip paths
        if self.config.skip_paths:
            if any(path.startswith(skip) for skip in self.config.skip_paths):
                return False
        
        # Check only paths
        if self.config.only_paths:
            if not any(path.startswith(only) for only in self.config.only_paths):
                return False
        
        return True
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Dispatch middleware.
        
        Args:
            request: Request object
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        if not self.config.enabled:
            return await call_next(request)
        
        path = request.url.path
        if not self._should_process(path):
            return await call_next(request)
        
        start_time = time.time()
        error = None
        response = None
        
        try:
            # Process request handlers
            for handler in self.request_handlers:
                if asyncio.iscoroutinefunction(handler):
                    await handler(request)
                else:
                    handler(request)
            
            # Call next middleware/handler
            response = await call_next(request)
            
            # Process response handlers
            for handler in self.response_handlers:
                if asyncio.iscoroutinefunction(handler):
                    response = await handler(request, response) or response
                else:
                    response = handler(request, response) or response
            
            return response
            
        except Exception as e:
            error = e
            # Process error handlers
            for handler in self.error_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(request, e)
                    else:
                        handler(request, e)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}")
            
            # Re-raise if not handled
            raise
            
        finally:
            # Process finally handlers
            duration = time.time() - start_time
            for handler in self.finally_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(request, response, duration, error)
                    else:
                        handler(request, response, duration, error)
                except Exception as handler_error:
                    logger.error(f"Finally handler failed: {handler_error}")


import asyncio


class MiddlewareManager:
    """Manager for multiple middlewares."""
    
    def __init__(self):
        """Initialize middleware manager."""
        self.middlewares: List[AdvancedMiddleware] = []
    
    def register(self, middleware: AdvancedMiddleware):
        """
        Register middleware.
        
        Args:
            middleware: Middleware instance
        """
        self.middlewares.append(middleware)
        # Sort by priority
        self.middlewares.sort(key=lambda m: m.config.priority, reverse=True)
    
    def create_timing_middleware(self) -> AdvancedMiddleware:
        """Create timing middleware."""
        middleware = AdvancedMiddleware(None)
        
        def timing_handler(request: Request, response: Response, duration: float, error: Optional[Exception]):
            if response:
                response.headers["X-Process-Time"] = f"{duration:.3f}"
        
        middleware.add_finally_handler(timing_handler)
        return middleware
    
    def create_logging_middleware(self) -> AdvancedMiddleware:
        """Create logging middleware."""
        middleware = AdvancedMiddleware(None)
        
        def request_handler(request: Request):
            logger.info(f"{request.method} {request.url.path}")
        
        def response_handler(request: Request, response: Response):
            logger.info(f"{request.method} {request.url.path} - {response.status_code}")
            return response
        
        middleware.add_request_handler(request_handler)
        middleware.add_response_handler(response_handler)
        return middleware



