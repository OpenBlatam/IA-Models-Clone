"""
Lazy Loader for Piel Mejorador AI SAM3
======================================

Lazy loading system for better startup performance.
"""

import logging
from typing import TypeVar, Callable, Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LazyProperty:
    """
    Lazy property descriptor.
    
    Caches the result after first access.
    """
    
    def __init__(self, func: Callable[[Any], T]):
        """
        Initialize lazy property.
        
        Args:
            func: Function to call lazily
        """
        self.func = func
        self.attr_name = f"_lazy_{func.__name__}"
        self.__doc__ = func.__doc__
    
    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T:
        """Get property value, computing if necessary."""
        if obj is None:
            return self
        
        if not hasattr(obj, self.attr_name):
            logger.debug(f"Lazy loading {self.func.__name__}")
            setattr(obj, self.attr_name, self.func(obj))
        
        return getattr(obj, self.attr_name)
    
    def __set__(self, obj: Any, value: T):
        """Set property value."""
        setattr(obj, self.attr_name, value)
    
    def __delete__(self, obj: Any):
        """Delete cached value."""
        if hasattr(obj, self.attr_name):
            delattr(obj, self.attr_name)


def lazy_property(func: Callable[[Any], T]) -> T:
    """
    Decorator for lazy properties.
    
    Example:
        class MyClass:
            @lazy_property
            def expensive_computation(self):
                # This is only computed once
                return expensive_operation()
    """
    return LazyProperty(func)


class LazyLoader:
    """
    Lazy loader for modules and classes.
    
    Features:
    - Deferred imports
    - On-demand loading
    - Caching
    """
    
    def __init__(self):
        """Initialize lazy loader."""
        self._cache: dict[str, Any] = {}
        self._loaders: dict[str, Callable] = {}
    
    def register(self, name: str, loader: Callable[[], Any]):
        """
        Register a lazy loader.
        
        Args:
            name: Resource name
            loader: Function that loads the resource
        """
        self._loaders[name] = loader
        logger.debug(f"Registered lazy loader: {name}")
    
    def get(self, name: str) -> Any:
        """
        Get resource, loading if necessary.
        
        Args:
            name: Resource name
            
        Returns:
            Loaded resource
        """
        if name in self._cache:
            return self._cache[name]
        
        if name not in self._loaders:
            raise ValueError(f"Lazy loader not found: {name}")
        
        logger.debug(f"Lazy loading: {name}")
        resource = self._loaders[name]()
        self._cache[name] = resource
        return resource
    
    def clear_cache(self, name: Optional[str] = None):
        """
        Clear cache.
        
        Args:
            name: Optional resource name (all if None)
        """
        if name:
            self._cache.pop(name, None)
        else:
            self._cache.clear()
    
    def preload(self, *names: str):
        """
        Preload resources.
        
        Args:
            *names: Resource names to preload
        """
        for name in names:
            self.get(name)




