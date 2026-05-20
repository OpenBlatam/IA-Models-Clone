"""
Middleware System - Chain of responsibility for processing
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class IMiddleware(ABC):
    """
    Interface for middleware
    """
    
    @abstractmethod
    def process(self, request: Any, next_handler: Callable) -> Any:
        """Process request and call next handler"""
        pass


class BaseMiddleware(IMiddleware):
    """
    Base middleware implementation
    """
    
    def __init__(self, name: str = "BaseMiddleware"):
        self.name = name
    
    def process(self, request: Any, next_handler: Callable) -> Any:
        """Base process - override in subclasses"""
        return next_handler(request)
    
    def before(self, request: Any) -> Any:
        """Called before processing - override if needed"""
        return request
    
    def after(self, response: Any) -> Any:
        """Called after processing - override if needed"""
        return response


class MiddlewarePipeline:
    """
    Pipeline for chaining middleware
    """
    
    def __init__(self):
        self.middlewares: list[IMiddleware] = []
    
    def add(self, middleware: IMiddleware) -> 'MiddlewarePipeline':
        """Add middleware to pipeline"""
        self.middlewares.append(middleware)
        logger.info(f"Added middleware: {middleware.name if hasattr(middleware, 'name') else type(middleware).__name__}")
        return self
    
    def remove(self, middleware: IMiddleware) -> 'MiddlewarePipeline':
        """Remove middleware from pipeline"""
        if middleware in self.middlewares:
            self.middlewares.remove(middleware)
        return self
    
    def execute(self, request: Any, final_handler: Callable) -> Any:
        """
        Execute pipeline with final handler
        
        Args:
            request: Request to process
            final_handler: Final handler to call
        
        Returns:
            Processed response
        """
        def build_chain(index: int) -> Callable:
            if index >= len(self.middlewares):
                return final_handler
            
            middleware = self.middlewares[index]
            
            def handler(req: Any) -> Any:
                return middleware.process(req, build_chain(index + 1))
            
            return handler
        
        return build_chain(0)(request)
    
    def clear(self):
        """Clear all middleware"""
        self.middlewares.clear()








