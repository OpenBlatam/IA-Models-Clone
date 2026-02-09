"""
Pipeline Utilities for Piel Mejorador AI SAM3
=============================================

Unified data transformation pipeline utilities.
"""

import logging
from typing import Callable, Any, List, Optional, TypeVar, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class PipelineStep:
    """Pipeline step definition."""
    name: str
    func: Callable[[Any], Any]
    error_handler: Optional[Callable[[Exception, Any], Any]] = None
    skip_on_error: bool = False


class PipelineUtils:
    """Unified pipeline utilities."""
    
    @staticmethod
    def create_pipeline(
        *steps: Callable[[Any], Any],
        error_handler: Optional[Callable[[Exception, Any], Any]] = None
    ) -> Callable[[Any], Any]:
        """
        Create a pipeline from functions.
        
        Args:
            *steps: Functions to chain
            error_handler: Optional error handler
            
        Returns:
            Pipeline function
        """
        def pipeline(data: Any) -> Any:
            result = data
            for i, step in enumerate(steps):
                try:
                    result = step(result)
                except Exception as e:
                    if error_handler:
                        result = error_handler(e, result)
                    else:
                        logger.error(f"Error in pipeline step {i}: {e}")
                        raise
            return result
        
        return pipeline
    
    @staticmethod
    def create_async_pipeline(
        *steps: Callable[[Any], Any],
        error_handler: Optional[Callable[[Exception, Any], Any]] = None
    ) -> Callable[[Any], Any]:
        """
        Create an async pipeline from functions.
        
        Args:
            *steps: Async functions to chain
            error_handler: Optional error handler
            
        Returns:
            Async pipeline function
        """
        async def pipeline(data: Any) -> Any:
            result = data
            for i, step in enumerate(steps):
                try:
                    import asyncio
                    if asyncio.iscoroutinefunction(step):
                        result = await step(result)
                    else:
                        result = step(result)
                except Exception as e:
                    if error_handler:
                        if asyncio.iscoroutinefunction(error_handler):
                            result = await error_handler(e, result)
                        else:
                            result = error_handler(e, result)
                    else:
                        logger.error(f"Error in pipeline step {i}: {e}")
                        raise
            return result
        
        return pipeline
    
    @staticmethod
    def transform(
        data: Any,
        *transformers: Callable[[Any], Any]
    ) -> Any:
        """
        Transform data through multiple transformers.
        
        Args:
            data: Data to transform
            *transformers: Transformer functions
            
        Returns:
            Transformed data
        """
        result = data
        for transformer in transformers:
            result = transformer(result)
        return result
    
    @staticmethod
    async def transform_async(
        data: Any,
        *transformers: Callable[[Any], Any]
    ) -> Any:
        """
        Transform data through multiple async transformers.
        
        Args:
            data: Data to transform
            *transformers: Transformer functions (can be async or sync)
            
        Returns:
            Transformed data
        """
        import asyncio
        
        result = data
        for transformer in transformers:
            if asyncio.iscoroutinefunction(transformer):
                result = await transformer(result)
            else:
                result = transformer(result)
        return result
    
    @staticmethod
    def filter_pipeline(
        data: List[Any],
        *filters: Callable[[Any], bool]
    ) -> List[Any]:
        """
        Filter data through multiple filters.
        
        Args:
            data: List of items to filter
            *filters: Filter functions (predicates)
            
        Returns:
            Filtered list
        """
        result = data
        for filter_func in filters:
            result = [item for item in result if filter_func(item)]
        return result
    
    @staticmethod
    def map_pipeline(
        data: List[Any],
        *mappers: Callable[[Any], Any]
    ) -> List[Any]:
        """
        Map data through multiple mappers.
        
        Args:
            data: List of items to map
            *mappers: Mapper functions
            
        Returns:
            Mapped list
        """
        result = data
        for mapper in mappers:
            result = [mapper(item) for item in result]
        return result
    
    @staticmethod
    def reduce_pipeline(
        data: List[Any],
        reducer: Callable[[Any, Any], Any],
        initial: Optional[Any] = None
    ) -> Any:
        """
        Reduce data through reducer function.
        
        Args:
            data: List of items to reduce
            reducer: Reducer function
            initial: Optional initial value
            
        Returns:
            Reduced value
        """
        if initial is not None:
            result = initial
            for item in data:
                result = reducer(result, item)
        else:
            if not data:
                raise ValueError("Cannot reduce empty list without initial value")
            result = data[0]
            for item in data[1:]:
                result = reducer(result, item)
        return result


# Convenience functions
def create_pipeline(*steps: Callable[[Any], Any], **kwargs) -> Callable[[Any], Any]:
    """Create pipeline."""
    return PipelineUtils.create_pipeline(*steps, **kwargs)


def transform(data: Any, *transformers: Callable[[Any], Any]) -> Any:
    """Transform data."""
    return PipelineUtils.transform(data, *transformers)


def filter_pipeline(data: List[Any], *filters: Callable[[Any], bool]) -> List[Any]:
    """Filter data."""
    return PipelineUtils.filter_pipeline(data, *filters)


def map_pipeline(data: List[Any], *mappers: Callable[[Any], Any]) -> List[Any]:
    """Map data."""
    return PipelineUtils.map_pipeline(data, *mappers)




