"""
Base Classes - Core Abstractions
================================

Core base classes and abstractions for the framework.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)


class BaseComponent(ABC):
    """
    Abstract base class for all framework components.
    
    Provides common functionality:
    - Configuration management
    - Device management
    - Logging
    - State management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize component.
        
        Args:
            config: Component configuration
        """
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize component (called after __init__)."""
        if not self._initialized:
            self._initialize()
            self._initialized = True
    
    @abstractmethod
    def _initialize(self) -> None:
        """Internal initialization (implemented by subclasses)."""
        pass
    
    def to_device(self, device: Optional[torch.device] = None) -> 'BaseComponent':
        """Move component to device."""
        if device is None:
            device = self.device
        self.device = device
        return self
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(config={self.config})"


class ComponentRegistry:
    """
    Registry for component registration and retrieval.
    
    Implements registry pattern for dynamic component management.
    """
    
    def __init__(self):
        """Initialize registry."""
        self._components: Dict[str, type] = {}
        self._instances: Dict[str, Any] = {}
    
    def register(self, name: str, component_class: type, singleton: bool = False) -> None:
        """
        Register a component class.
        
        Args:
            name: Component name
            component_class: Component class
            singleton: Whether to create singleton instances
        """
        self._components[name] = component_class
        if singleton:
            self._instances[name] = None
        logger.info(f"Registered component: {name}")
    
    def get_class(self, name: str) -> type:
        """
        Get component class by name.
        
        Args:
            name: Component name
            
        Returns:
            Component class
            
        Raises:
            KeyError: If component not found
        """
        if name not in self._components:
            raise KeyError(f"Component '{name}' not found in registry")
        return self._components[name]
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create component instance.
        
        Args:
            name: Component name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Component instance
        """
        component_class = self.get_class(name)
        instance = component_class(*args, **kwargs)
        
        # Handle singleton
        if name in self._instances and self._instances[name] is None:
            self._instances[name] = instance
        
        return instance
    
    def get_singleton(self, name: str) -> Optional[Any]:
        """
        Get singleton instance.
        
        Args:
            name: Component name
            
        Returns:
            Singleton instance or None
        """
        return self._instances.get(name)
    
    def list_components(self) -> list:
        """List all registered components."""
        return list(self._components.keys())


class Factory:
    """
    Generic factory for creating instances.
    
    Provides factory pattern implementation with validation.
    """
    
    def __init__(self, registry: Optional[ComponentRegistry] = None):
        """
        Initialize factory.
        
        Args:
            registry: Component registry (creates new if None)
        """
        self.registry = registry or ComponentRegistry()
    
    def register(self, name: str, component_class: type) -> None:
        """Register component."""
        self.registry.register(name, component_class)
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create instance.
        
        Args:
            name: Component name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Component instance
        """
        return self.registry.create(name, *args, **kwargs)
    
    def create_with_config(self, name: str, config: Dict[str, Any]) -> Any:
        """
        Create instance with configuration.
        
        Args:
            name: Component name
            config: Configuration dictionary
            
        Returns:
            Component instance
        """
        return self.registry.create(name, config=config)



