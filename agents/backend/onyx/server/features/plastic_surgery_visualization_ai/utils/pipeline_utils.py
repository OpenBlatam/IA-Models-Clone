"""Pipeline utilities."""

from typing import List, Callable, Any, TypeVar, Optional
from functools import reduce

T = TypeVar('T')
R = TypeVar('R')


class Pipeline:
    """Data processing pipeline."""
    
    def __init__(self, steps: Optional[List[Callable]] = None):
        self._steps = steps or []
    
    def add_step(self, step: Callable) -> 'Pipeline':
        """
        Add processing step.
        
        Args:
            step: Processing function
            
        Returns:
            Self for chaining
        """
        self._steps.append(step)
        return self
    
    def execute(self, data: T) -> Any:
        """
        Execute pipeline on data.
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        return reduce(lambda acc, step: step(acc), self._steps, data)
    
    async def execute_async(self, data: T) -> Any:
        """
        Execute pipeline asynchronously.
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        import asyncio
        
        result = data
        for step in self._steps:
            if asyncio.iscoroutinefunction(step):
                result = await step(result)
            else:
                result = step(result)
        return result


class FilterPipeline:
    """Filter pipeline."""
    
    def __init__(self, filters: Optional[List[Callable]] = None):
        self._filters = filters or []
    
    def add_filter(self, filter_func: Callable[[Any], bool]) -> 'FilterPipeline':
        """
        Add filter.
        
        Args:
            filter_func: Filter function
            
        Returns:
            Self for chaining
        """
        self._filters.append(filter_func)
        return self
    
    def apply(self, items: List[T]) -> List[T]:
        """
        Apply filters to items.
        
        Args:
            items: List of items
            
        Returns:
            Filtered list
        """
        result = items
        for filter_func in self._filters:
            result = [item for item in result if filter_func(item)]
        return result


class TransformPipeline:
    """Transform pipeline."""
    
    def __init__(self, transforms: Optional[List[Callable]] = None):
        self._transforms = transforms or []
    
    def add_transform(self, transform: Callable) -> 'TransformPipeline':
        """
        Add transform.
        
        Args:
            transform: Transform function
            
        Returns:
            Self for chaining
        """
        self._transforms.append(transform)
        return self
    
    def apply(self, data: T) -> Any:
        """
        Apply transforms to data.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        result = data
        for transform in self._transforms:
            result = transform(result)
        return result

