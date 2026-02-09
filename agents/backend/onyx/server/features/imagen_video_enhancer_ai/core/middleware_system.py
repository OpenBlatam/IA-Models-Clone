"""
Middleware System
=================

Advanced middleware system for request/response processing.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MiddlewareType(Enum):
    """Middleware type."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    FINALLY = "finally"


@dataclass
class RequestContext:
    """Request context for middleware."""
    path: str
    method: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Any] = None
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Get request duration in milliseconds."""
        return (time.time() - self.start_time) * 1000


@dataclass
class ResponseContext:
    """Response context for middleware."""
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Middleware:
    """Base middleware class."""
    
    def __init__(self, name: str, middleware_type: MiddlewareType):
        """
        Initialize middleware.
        
        Args:
            name: Middleware name
            middleware_type: Middleware type
        """
        self.name = name
        self.middleware_type = middleware_type
        self.enabled = True
        self.priority = 0  # Lower priority runs first
    
    async def process(
        self,
        context: RequestContext,
        next_handler: Optional[Callable[[RequestContext], Awaitable[ResponseContext]]] = None
    ) -> ResponseContext:
        """
        Process middleware.
        
        Args:
            context: Request context
            next_handler: Next handler in chain
            
        Returns:
            Response context
        """
        if not self.enabled:
            if next_handler:
                return await next_handler(context)
            return ResponseContext(status_code=200)
        
        return await self._process(context, next_handler)
    
    async def _process(
        self,
        context: RequestContext,
        next_handler: Optional[Callable[[RequestContext], Awaitable[ResponseContext]]]
    ) -> ResponseContext:
        """
        Subclass-specific processing.
        
        Args:
            context: Request context
            next_handler: Next handler
            
        Returns:
            Response context
        """
        if next_handler:
            return await next_handler(context)
        return ResponseContext(status_code=200)
    
    def __lt__(self, other):
        """Compare by priority."""
        return self.priority < other.priority


class MiddlewarePipeline:
    """Middleware pipeline for processing requests."""
    
    def __init__(self):
        """Initialize middleware pipeline."""
        self.middlewares: Dict[MiddlewareType, List[Middleware]] = {
            MiddlewareType.REQUEST: [],
            MiddlewareType.RESPONSE: [],
            MiddlewareType.ERROR: [],
            MiddlewareType.FINALLY: []
        }
    
    def add(self, middleware: Middleware):
        """
        Add middleware to pipeline.
        
        Args:
            middleware: Middleware instance
        """
        self.middlewares[middleware.middleware_type].append(middleware)
        self.middlewares[middleware.middleware_type].sort()
        logger.info(f"Added middleware: {middleware.name} ({middleware.middleware_type.value})")
    
    async def process(
        self,
        context: RequestContext,
        handler: Callable[[RequestContext], Awaitable[ResponseContext]]
    ) -> ResponseContext:
        """
        Process request through middleware pipeline.
        
        Args:
            context: Request context
            handler: Final request handler
            
        Returns:
            Response context
        """
        # Build middleware chain recursively
        async def execute_chain(middlewares: List[Middleware], index: int, next_h: Callable):
            if index >= len(middlewares):
                return await next_h(context)
            
            middleware = middlewares[index]
            async def current_handler(ctx: RequestContext):
                return await execute_chain(middlewares, index + 1, next_h)
            
            return await middleware.process(context, current_handler)
        
        # Combine request and response middlewares
        request_mws = self.middlewares[MiddlewareType.REQUEST]
        response_mws = self.middlewares[MiddlewareType.RESPONSE]
        
        # Execute chain
        try:
            # Process request middlewares
            async def process_with_response_mw(ctx: RequestContext):
                # Process response middlewares
                async def process_handler(ctx: RequestContext):
                    return await handler(ctx)
                
                return await execute_chain(response_mws, 0, process_handler)
            
            response = await execute_chain(request_mws, 0, process_with_response_mw)
            
            # Execute finally middlewares
            for middleware in self.middlewares[MiddlewareType.FINALLY]:
                await middleware.process(context)
            
            return response
        
        except Exception as e:
            # Execute error middlewares
            for middleware in self.middlewares[MiddlewareType.ERROR]:
                try:
                    await middleware.process(context)
                except Exception as error_mw_error:
                    logger.error(f"Error in error middleware {middleware.name}: {error_mw_error}")
            
            raise

