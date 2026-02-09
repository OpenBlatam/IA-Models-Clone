"""
Component Registry - Generic registry for component classes.

Single Responsibility: Register and retrieve component classes.
This eliminates duplication across factories.
"""

from __future__ import annotations

from typing import Dict, Type, TypeVar, Protocol, Optional

T = TypeVar('T', bound=Protocol)


class ComponentRegistry:
    """
    Generic registry for audio components.
    
    Single Responsibility: Maintain a registry of component classes.
    Used by all factories to eliminate duplicate registration code.
    
    Example:
        registry = ComponentRegistry()
        registry.register("spleeter", SpleeterSeparator)
        separator_class = registry.get("spleeter")
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._components: Dict[str, Type[T]] = {}
    
    def register(self, name: str, component_class: Type[T]) -> None:
        """
        Register a component class.
        
        Args:
            name: Name of the component (case-insensitive)
            component_class: Class to register
            
        Raises:
            TypeError: If component_class is not a class
        """
        if not isinstance(component_class, type):
            raise TypeError(f"component_class must be a class, got {type(component_class)}")
        
        self._components[name.lower()] = component_class
    
    def get(self, name: str) -> Type[T]:
        """
        Get a registered component class.
        
        Args:
            name: Name of the component (case-insensitive)
            
        Returns:
            Registered component class
            
        Raises:
            KeyError: If component is not registered
        """
        name_lower = name.lower()
        if name_lower not in self._components:
            raise KeyError(
                f"Component '{name}' not registered. "
                f"Available: {list(self._components.keys())}"
            )
        return self._components[name_lower]
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a component is registered.
        
        Args:
            name: Name of the component (case-insensitive)
            
        Returns:
            True if registered, False otherwise
        """
        return name.lower() in self._components
    
    def list_registered(self) -> list[str]:
        """
        List all registered components.
        
        Returns:
            List of registered component names
        """
        return list(self._components.keys())
    
    def unregister(self, name: str) -> None:
        """
        Unregister a component.
        
        Args:
            name: Name of the component to unregister
        """
        self._components.pop(name.lower(), None)
    
    def clear(self) -> None:
        """Clear all registered components."""
        self._components.clear()

