"""Adapter pattern utilities."""

from typing import Callable, TypeVar, Protocol, runtime_checkable

T = TypeVar('T')
R = TypeVar('R')


@runtime_checkable
class Adaptable(Protocol):
    """Protocol for adaptable objects."""
    
    def adapt(self, target_type: type) -> Any:
        """Adapt to target type."""
        ...


class Adapter:
    """Generic adapter."""
    
    def __init__(self, adaptee: Any):
        self._adaptee = adaptee
    
    def adapt(self, target_type: type) -> Any:
        """
        Adapt adaptee to target type.
        
        Args:
            target_type: Target type
            
        Returns:
            Adapted object
        """
        if isinstance(self._adaptee, target_type):
            return self._adaptee
        
        # Try to find adapter method
        adapter_method = f"_adapt_to_{target_type.__name__}"
        if hasattr(self, adapter_method):
            return getattr(self, adapter_method)(self._adaptee)
        
        raise ValueError(f"Cannot adapt {type(self._adaptee)} to {target_type}")


class FunctionAdapter:
    """Adapter using function."""
    
    def __init__(self, adapt_func: Callable[[T], R]):
        self._adapt_func = adapt_func
    
    def adapt(self, value: T) -> R:
        """
        Adapt value using function.
        
        Args:
            value: Value to adapt
            
        Returns:
            Adapted value
        """
        return self._adapt_func(value)


class TypeAdapter:
    """Type-based adapter registry."""
    
    def __init__(self):
        self._adapters: dict = {}
    
    def register(self, source_type: type, target_type: type, adapter: Callable) -> None:
        """
        Register adapter.
        
        Args:
            source_type: Source type
            target_type: Target type
            adapter: Adapter function
        """
        key = (source_type, target_type)
        self._adapters[key] = adapter
    
    def adapt(self, value: Any, target_type: type) -> Any:
        """
        Adapt value to target type.
        
        Args:
            value: Value to adapt
            target_type: Target type
            
        Returns:
            Adapted value
        """
        source_type = type(value)
        
        if isinstance(value, target_type):
            return value
        
        key = (source_type, target_type)
        if key in self._adapters:
            return self._adapters[key](value)
        
        raise ValueError(f"No adapter from {source_type} to {target_type}")



