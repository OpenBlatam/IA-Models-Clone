"""
Pipeline Builder - Build processing pipelines
"""

from typing import List, Callable, Any
import logging

from ..middleware.middleware import MiddlewarePipeline, IMiddleware

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Builder for creating processing pipelines
    """
    
    def __init__(self):
        self._middlewares: List[IMiddleware] = []
        self._final_handler: Optional[Callable] = None
    
    def add_middleware(self, middleware: IMiddleware) -> 'PipelineBuilder':
        """Add middleware to pipeline"""
        self._middlewares.append(middleware)
        return self
    
    def with_final_handler(self, handler: Callable) -> 'PipelineBuilder':
        """Set final handler"""
        self._final_handler = handler
        return self
    
    def build(self) -> MiddlewarePipeline:
        """Build pipeline"""
        pipeline = MiddlewarePipeline()
        
        for middleware in self._middlewares:
            pipeline.add(middleware)
        
        if self._final_handler:
            # Store handler for later use
            pipeline._final_handler = self._final_handler
        
        return pipeline
    
    def execute(self, request: Any) -> Any:
        """Build and execute pipeline"""
        if self._final_handler is None:
            raise ValueError("Final handler not set")
        
        pipeline = self.build()
        return pipeline.execute(request, self._final_handler)


# Convenience function
def build_pipeline() -> PipelineBuilder:
    """Start building a pipeline"""
    return PipelineBuilder()








