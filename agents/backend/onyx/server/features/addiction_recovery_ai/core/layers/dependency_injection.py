"""
Dependency Injection System - Ultra Modular Component Management
Enables loose coupling and testability through dependency injection
"""

from typing import Optional, Dict, Any, List, Type, Callable, get_type_hints
import logging
from functools import wraps

logger = logging.getLogger(__name__)


# ============================================================================
# Dependency Container
# ============================================================================

class DependencyContainer:
    """Dependency injection container"""
    
    def __init__(self):
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._types: Dict[Type, str] = {}
    
    def register_singleton(self, name: str, instance: Any):
        """Register singleton instance"""
        self._singletons[name] = instance
        logger.debug(f"Registered singleton: {name}")
    
    def register_factory(self, name: str, factory: Callable, *args, **kwargs):
        """Register factory function"""
        def factory_wrapper():
            return factory(*args, **kwargs)
        self._factories[name] = factory_wrapper
        logger.debug(f"Registered factory: {name}")
    
    def register_type(self, interface_type: Type, implementation_name: str):
        """Register type mapping"""
        self._types[interface_type] = implementation_name
        logger.debug(f"Registered type: {interface_type.__name__} -> {implementation_name}")
    
    def get(self, name: str) -> Any:
        """Get dependency by name"""
        # Check singletons
        if name in self._singletons:
            return self._singletons[name]
        
        # Check factories
        if name in self._factories:
            instance = self._factories[name]()
            # Cache as singleton
            self._singletons[name] = instance
            return instance
        
        raise ValueError(f"Dependency '{name}' not found")
    
    def get_by_type(self, interface_type: Type) -> Any:
        """Get dependency by type"""
        if interface_type in self._types:
            return self.get(self._types[interface_type])
        raise ValueError(f"Dependency for type '{interface_type.__name__}' not found")
    
    def inject(self, func: Callable) -> Callable:
        """Decorator to inject dependencies into function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get type hints
            hints = get_type_hints(func)
            
            # Inject dependencies
            for param_name, param_type in hints.items():
                if param_name not in kwargs and param_name != 'return':
                    try:
                        kwargs[param_name] = self.get_by_type(param_type)
                    except ValueError:
                        pass  # Dependency not registered, use default
            
            return func(*args, **kwargs)
        
        return wrapper
    
    def clear(self):
        """Clear all dependencies"""
        self._singletons.clear()
        self._factories.clear()
        self._types.clear()


# ============================================================================
# Global Container Instance
# ============================================================================

_global_container = DependencyContainer()


def get_container() -> DependencyContainer:
    """Get global dependency container"""
    return _global_container


def reset_container():
    """Reset global container (useful for testing)"""
    global _global_container
    _global_container = DependencyContainer()


# ============================================================================
# Decorators
# ============================================================================

def inject_dependencies(func: Callable) -> Callable:
    """Decorator to inject dependencies"""
    return _global_container.inject(func)


def register_service(name: str, singleton: bool = True):
    """Decorator to register service"""
    def decorator(cls: Type):
        if singleton:
            instance = cls()
            _global_container.register_singleton(name, instance)
        else:
            _global_container.register_factory(name, cls)
        return cls
    return decorator


# Export main components
__all__ = [
    "DependencyContainer",
    "get_container",
    "reset_container",
    "inject_dependencies",
    "register_service",
]



