"""
Pipeline Middleware

Middleware for processing pipelines.
"""

import logging
from typing import Callable, List, Any, Optional

logger = logging.getLogger(__name__)


class PipelineMiddleware:
    """Middleware for pipeline processing."""
    
    def __init__(
        self,
        name: str,
        process_fn: Callable
    ):
        """
        Initialize middleware.
        
        Args:
            name: Middleware name
            process_fn: Processing function
        """
        self.name = name
        self.process_fn = process_fn
    
    def __call__(
        self,
        data: Any,
        next_fn: Optional[Callable] = None
    ) -> Any:
        """
        Process data.
        
        Args:
            data: Input data
            next_fn: Next middleware or final handler
            
        Returns:
            Processed data
        """
        # Pre-processing
        processed_data = self.process_fn(data)
        
        # Call next middleware or handler
        if next_fn:
            return next_fn(processed_data)
        
        return processed_data


class MiddlewareChain:
    """Chain of middleware."""
    
    def __init__(self, middlewares: List[PipelineMiddleware]):
        """
        Initialize middleware chain.
        
        Args:
            middlewares: List of middleware
        """
        self.middlewares = middlewares
    
    def __call__(self, data: Any, handler: Callable) -> Any:
        """
        Process data through middleware chain.
        
        Args:
            data: Input data
            handler: Final handler function
            
        Returns:
            Processed data
        """
        # Build chain from end to start
        current = handler
        
        for middleware in reversed(self.middlewares):
            current = lambda d, m=middleware, n=current: m(d, n)
        
        return current(data)


def create_middleware_chain(
    *middlewares: PipelineMiddleware
) -> MiddlewareChain:
    """
    Create middleware chain.
    
    Args:
        *middlewares: Middleware instances
        
    Returns:
        MiddlewareChain instance
    """
    return MiddlewareChain(list(middlewares))


def apply_middleware(
    data: Any,
    handler: Callable,
    *middlewares: PipelineMiddleware
) -> Any:
    """
    Apply middleware to data.
    
    Args:
        data: Input data
        handler: Handler function
        *middlewares: Middleware instances
        
    Returns:
        Processed data
    """
    chain = create_middleware_chain(*middlewares)
    return chain(data, handler)



