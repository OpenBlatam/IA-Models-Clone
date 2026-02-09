"""
Core Container
===============

Simple and clear dependency injection container for the Document Workflow Chain system.
"""

from __future__ import annotations
from typing import Dict, Any, TypeVar, Type, Optional
import inspect

T = TypeVar('T')


class Container:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(self, service_type: Type[T], instance: T, singleton: bool = False) -> None:
        """Register a service instance"""
        key = service_type.__name__
        
        if singleton:
            self._singletons[key] = instance
        else:
            self._services[key] = instance
    
    def register_factory(self, service_type: Type[T], factory: callable, singleton: bool = False) -> None:
        """Register a service factory"""
        key = service_type.__name__
        
        if singleton:
            self._singletons[key] = factory
        else:
            self._services[key] = factory
    
    def get(self, service_type: Type[T]) -> T:
        """Get a service instance"""
        key = service_type.__name__
        
        # Check singletons first
        if key in self._singletons:
            service = self._singletons[key]
            if callable(service):
                self._singletons[key] = service()
            return self._singletons[key]
        
        # Check regular services
        if key in self._services:
            service = self._services[key]
            if callable(service):
                return service()
            return service
        
        raise ValueError(f"Service {key} not found")
    
    def get_optional(self, service_type: Type[T]) -> Optional[T]:
        """Get a service instance or None"""
        try:
            return self.get(service_type)
        except ValueError:
            return None
    
    def clear(self) -> None:
        """Clear all services"""
        self._services.clear()
        self._singletons.clear()


# Global container instance
container = Container()


