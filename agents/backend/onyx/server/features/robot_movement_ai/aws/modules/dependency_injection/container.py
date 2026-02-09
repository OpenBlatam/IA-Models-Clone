"""
Dependency Injection Container
===============================

Advanced DI container for managing dependencies.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, TypeVar
from functools import lru_cache

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DIContainer:
    """Dependency injection container."""
    
    def __init__(self):
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._transients: Dict[str, Type] = {}
        self._scoped: Dict[str, Any] = {}
        self._current_scope: Optional[str] = None
    
    def register_singleton(self, interface: Type[T], implementation: T, name: Optional[str] = None):
        """Register singleton instance."""
        key = name or interface.__name__
        self._singletons[key] = implementation
        logger.debug(f"Registered singleton: {key}")
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T], name: Optional[str] = None):
        """Register factory function."""
        key = name or interface.__name__
        self._factories[key] = factory
        logger.debug(f"Registered factory: {key}")
    
    def register_transient(self, interface: Type[T], implementation: Type[T], name: Optional[str] = None):
        """Register transient (new instance each time)."""
        key = name or interface.__name__
        self._transients[key] = implementation
        logger.debug(f"Registered transient: {key}")
    
    def register_scoped(self, interface: Type[T], implementation: Type[T], name: Optional[str] = None):
        """Register scoped (one instance per scope)."""
        key = name or interface.__name__
        self._transients[key] = implementation  # Will be scoped
        logger.debug(f"Registered scoped: {key}")
    
    def get(self, interface: Type[T], name: Optional[str] = None) -> Optional[T]:
        """Get dependency instance."""
        key = name or interface.__name__
        
        # Check singletons
        if key in self._singletons:
            return self._singletons[key]
        
        # Check factories
        if key in self._factories:
            instance = self._factories[key]()
            # Cache as singleton if not already cached
            if key not in self._singletons:
                self._singletons[key] = instance
            return instance
        
        # Check scoped
        if self._current_scope and key in self._scoped.get(self._current_scope, {}):
            return self._scoped[self._current_scope][key]
        
        # Check transients
        if key in self._transients:
            return self._transients[key]()
        
        return None
    
    def create_scope(self, scope_name: str) -> "Scope":
        """Create a new scope."""
        return Scope(self, scope_name)
    
    def clear(self):
        """Clear all registrations."""
        self._singletons.clear()
        self._factories.clear()
        self._transients.clear()
        self._scoped.clear()


class Scope:
    """Dependency injection scope."""
    
    def __init__(self, container: DIContainer, name: str):
        self.container = container
        self.name = name
        self.container._current_scope = name
        if name not in self.container._scoped:
            self.container._scoped[name] = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container._current_scope = None
    
    def get(self, interface: Type[T], name: Optional[str] = None) -> Optional[T]:
        """Get scoped dependency."""
        key = name or interface.__name__
        
        # Check if already in scope
        if key in self.container._scoped[self.name]:
            return self.container._scoped[self.name][key]
        
        # Create new instance
        if key in self.container._transients:
            instance = self.container._transients[key]()
            self.container._scoped[self.name][key] = instance
            return instance
        
        return None


# Global container
_global_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Get global DI container."""
    global _global_container
    if _global_container is None:
        _global_container = DIContainer()
    return _global_container















