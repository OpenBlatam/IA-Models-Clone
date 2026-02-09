"""Lazy evaluation utilities."""

from typing import Callable, TypeVar, Optional, Any
from functools import wraps

T = TypeVar('T')


class LazyValue:
    """Lazy value that computes on first access."""
    
    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._value: Optional[T] = None
        self._computed = False
    
    def get(self) -> T:
        """Get value, computing if necessary."""
        if not self._computed:
            self._value = self._factory()
            self._computed = True
        return self._value
    
    def reset(self) -> None:
        """Reset to recompute on next access."""
        self._computed = False
        self._value = None
    
    @property
    def is_computed(self) -> bool:
        """Check if value is computed."""
        return self._computed


def lazy_property(func: Callable) -> property:
    """Decorator for lazy property."""
    attr_name = f'_lazy_{func.__name__}'
    
    @property
    @wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return wrapper


class LazyDict:
    """Dictionary with lazy value computation."""
    
    def __init__(self):
        self._data: dict = {}
        self._factories: dict = {}
    
    def set(self, key: str, value: Any) -> None:
        """Set immediate value."""
        self._data[key] = value
        self._factories.pop(key, None)
    
    def set_lazy(self, key: str, factory: Callable[[], Any]) -> None:
        """Set lazy value."""
        self._factories[key] = factory
        self._data.pop(key, None)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value, computing if lazy."""
        if key in self._data:
            return self._data[key]
        
        if key in self._factories:
            value = self._factories[key]()
            self._data[key] = value
            self._factories.pop(key)
            return value
        
        return default
    
    def __getitem__(self, key: str) -> Any:
        """Get item."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set item."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data or key in self._factories

