"""
Middleware Base
===============

Base middleware pattern for request/response processing.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class Request:
    """Request object."""
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    body: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Response:
    """Response object."""
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseMiddleware(ABC):
    """Base middleware interface."""
    
    def __init__(self, name: str = "Middleware"):
        """
        Initialize middleware.
        
        Args:
            name: Middleware name
        """
        self.name = name
        self.enabled = True
    
    @abstractmethod
    async def process_request(self, request: Request) -> Optional[Response]:
        """
        Process request.
        
        Args:
            request: Request object
            
        Returns:
            Response if should stop processing, None to continue
        """
        pass
    
    @abstractmethod
    async def process_response(self, request: Request, response: Response) -> Response:
        """
        Process response.
        
        Args:
            request: Original request
            response: Response object
            
        Returns:
            Modified response
        """
        pass
    
    async def process_error(self, request: Request, error: Exception) -> Response:
        """
        Process error.
        
        Args:
            request: Original request
            error: Exception that occurred
            
        Returns:
            Error response
        """
        logger.error(f"Error in {self.name}: {error}")
        return Response(
            status_code=500,
            body={"error": str(error)}
        )


class MiddlewarePipeline:
    """Middleware pipeline for request/response processing."""
    
    def __init__(self):
        """Initialize middleware pipeline."""
        self.middlewares: List[BaseMiddleware] = []
    
    def add(self, middleware: BaseMiddleware):
        """
        Add middleware to pipeline.
        
        Args:
            middleware: Middleware instance
        """
        self.middlewares.append(middleware)
        logger.debug(f"Added middleware: {middleware.name}")
    
    async def process(
        self,
        request: Request,
        handler: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Process request through pipeline.
        
        Args:
            request: Request object
            handler: Final request handler
            
        Returns:
            Response object
        """
        # Process request through middlewares
        for middleware in self.middlewares:
            if not middleware.enabled:
                continue
            
            try:
                response = await middleware.process_request(request)
                if response is not None:
                    # Middleware returned response, stop processing
                    return response
            except Exception as e:
                return await middleware.process_error(request, e)
        
        # Execute handler
        try:
            response = await handler(request)
        except Exception as e:
            # Process error through middlewares in reverse
            for middleware in reversed(self.middlewares):
                if middleware.enabled:
                    try:
                        return await middleware.process_error(request, e)
                    except Exception:
                        pass
            
            # Default error response
            return Response(
                status_code=500,
                body={"error": str(e)}
            )
        
        # Process response through middlewares in reverse
        for middleware in reversed(self.middlewares):
            if not middleware.enabled:
                continue
            
            try:
                response = await middleware.process_response(request, response)
            except Exception as e:
                logger.error(f"Error in middleware {middleware.name}: {e}")
        
        return response




