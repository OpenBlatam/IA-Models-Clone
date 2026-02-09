"""
Factory Module - Factory pattern implementations.

Provides:
- Factory functions for common objects
- Object creation helpers
- Builder patterns
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Factory:
    """Generic factory class."""
    
    def __init__(self):
        """Initialize factory."""
        self.creators: Dict[str, Callable] = {}
    
    def register(self, name: str, creator: Callable) -> None:
        """
        Register a creator function.
        
        Args:
            name: Object name
            creator: Creator function
        """
        self.creators[name] = creator
        logger.info(f"Registered factory creator: {name}")
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create object by name.
        
        Args:
            name: Object name
            *args: Creator arguments
            **kwargs: Creator keyword arguments
            
        Returns:
            Created object
        """
        if name not in self.creators:
            raise ValueError(f"Unknown factory type: {name}")
        
        return self.creators[name](*args, **kwargs)
    
    def list_types(self) -> List[str]:
        """List available types."""
        return list(self.creators.keys())


# Global factories
benchmark_factory = Factory()
model_factory = Factory()
backend_factory = Factory()


def create_benchmark(benchmark_type: str, *args, **kwargs):
    """Create benchmark instance."""
    return benchmark_factory.create(benchmark_type, *args, **kwargs)


def create_model(model_type: str, *args, **kwargs):
    """Create model instance."""
    return model_factory.create(model_type, *args, **kwargs)


def create_backend(backend_type: str, *args, **kwargs):
    """Create backend instance."""
    return backend_factory.create(backend_type, *args, **kwargs)












