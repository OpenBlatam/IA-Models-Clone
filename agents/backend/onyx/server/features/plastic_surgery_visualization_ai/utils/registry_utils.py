"""Registry pattern utilities."""

from typing import Dict, TypeVar, Callable, Optional, Any
import threading

T = TypeVar('T')


class Registry:
    """Simple registry."""
    
    def __init__(self):
        self._items: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, item: Any) -> None:
        """
        Register item.
        
        Args:
            name: Item name
            item: Item to register
        """
        with self._lock:
            self._items[name] = item
    
    def unregister(self, name: str) -> None:
        """
        Unregister item.
        
        Args:
            name: Item name
        """
        with self._lock:
            self._items.pop(name, None)
    
    def get(self, name: str, default: Any = None) -> Any:
        """
        Get item.
        
        Args:
            name: Item name
            default: Default value
            
        Returns:
            Registered item
        """
        return self._items.get(name, default)
    
    def has(self, name: str) -> bool:
        """
        Check if item is registered.
        
        Args:
            name: Item name
            
        Returns:
            True if registered
        """
        return name in self._items
    
    def list(self) -> list:
        """
        List all registered names.
        
        Returns:
            List of names
        """
        return list(self._items.keys())
    
    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._items.clear()


class FactoryRegistry:
    """Registry with factory functions."""
    
    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def register_factory(self, name: str, factory: Callable) -> None:
        """
        Register factory function.
        
        Args:
            name: Factory name
            factory: Factory function
        """
        with self._lock:
            self._factories[name] = factory
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create instance using factory.
        
        Args:
            name: Factory name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Created instance
        """
        if name not in self._factories:
            raise ValueError(f"Factory '{name}' not registered")
        
        factory = self._factories[name]
        return factory(*args, **kwargs)
    
    def get_or_create(self, name: str, *args, **kwargs) -> Any:
        """
        Get existing instance or create new one.
        
        Args:
            name: Factory name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Instance
        """
        if name in self._instances:
            return self._instances[name]
        
        instance = self.create(name, *args, **kwargs)
        with self._lock:
            self._instances[name] = instance
        return instance


def register(name: str, registry: Optional[Registry] = None) -> Callable:
    """
    Decorator to register item in registry.
    
    Args:
        name: Registration name
        registry: Registry instance (creates global if None)
        
    Returns:
        Decorator function
    """
    if registry is None:
        registry = _global_registry
    
    def decorator(item: Any) -> Any:
        registry.register(name, item)
        return item
    
    return decorator


# Global registry
_global_registry = Registry()



