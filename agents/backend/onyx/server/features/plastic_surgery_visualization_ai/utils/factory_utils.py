"""Factory pattern utilities."""

from typing import Dict, Callable, TypeVar, Optional, Any
import threading

T = TypeVar('T')


class Factory:
    """Generic factory."""
    
    def __init__(self):
        self._creators: Dict[str, Callable] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, creator: Callable) -> None:
        """
        Register creator function.
        
        Args:
            name: Factory name
            creator: Creator function
        """
        with self._lock:
            self._creators[name] = creator
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create instance.
        
        Args:
            name: Factory name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Created instance
        """
        if name not in self._creators:
            raise ValueError(f"Factory '{name}' not registered")
        
        creator = self._creators[name]
        return creator(*args, **kwargs)
    
    def has(self, name: str) -> bool:
        """
        Check if factory is registered.
        
        Args:
            name: Factory name
            
        Returns:
            True if registered
        """
        return name in self._creators


class SingletonFactory:
    """Factory that returns singletons."""
    
    def __init__(self):
        self._factory = Factory()
        self._instances: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, creator: Callable) -> None:
        """Register creator function."""
        self._factory.register(name, creator)
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create or get singleton instance.
        
        Args:
            name: Factory name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Singleton instance
        """
        if name in self._instances:
            return self._instances[name]
        
        with self._lock:
            if name not in self._instances:
                instance = self._factory.create(name, *args, **kwargs)
                self._instances[name] = instance
            return self._instances[name]


class CachedFactory:
    """Factory with caching."""
    
    def __init__(self, factory: Factory, cache_size: int = 100):
        self._factory = factory
        self._cache: Dict[str, Any] = {}
        self._cache_size = cache_size
        self._lock = threading.Lock()
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create instance with caching.
        
        Args:
            name: Factory name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Created instance
        """
        # Create cache key from name and arguments
        cache_key = f"{name}:{str(args)}:{str(kwargs)}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        instance = self._factory.create(name, *args, **kwargs)
        
        with self._lock:
            if len(self._cache) >= self._cache_size:
                # Remove oldest entry (simple FIFO)
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = instance
        
        return instance
    
    def clear_cache(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()


def factory(name: str, factory_instance: Optional[Factory] = None) -> Callable:
    """
    Decorator to register factory creator.
    
    Args:
        name: Factory name
        factory_instance: Factory instance (creates global if None)
        
    Returns:
        Decorator function
    """
    if factory_instance is None:
        factory_instance = _global_factory
    
    def decorator(creator: Callable) -> Callable:
        factory_instance.register(name, creator)
        return creator
    
    return decorator


# Global factory
_global_factory = Factory()



