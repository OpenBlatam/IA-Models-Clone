"""
Registry Base
=============

Base registry pattern for all registry types.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Generic, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRegistry(ABC, Generic[T]):
    """Base registry interface."""
    
    def __init__(self, name: str = "Registry"):
        """
        Initialize registry.
        
        Args:
            name: Registry name
        """
        self.name = name
        self.items: Dict[str, T] = {}
        self.item_types: Dict[str, Type[T]] = {}
    
    def register_type(self, name: str, item_type: Type[T]):
        """
        Register an item type.
        
        Args:
            name: Item name
            item_type: Item class
        """
        self.item_types[name] = item_type
        logger.debug(f"Registered type {name} in {self.name}")
    
    def create(self, name: str, *args, **kwargs) -> Optional[T]:
        """
        Create an item instance.
        
        Args:
            name: Item name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Item instance or None
        """
        if name not in self.item_types:
            logger.error(f"Type {name} not found in {self.name}")
            return None
        
        item_type = self.item_types[name]
        item = item_type(*args, **kwargs)
        self.items[name] = item
        return item
    
    def register(self, name: str, item: T):
        """
        Register an item instance.
        
        Args:
            name: Item name
            item: Item instance
        """
        self.items[name] = item
        logger.debug(f"Registered {name} in {self.name}")
    
    def get(self, name: str) -> Optional[T]:
        """
        Get item by name.
        
        Args:
            name: Item name
            
        Returns:
            Item instance or None
        """
        return self.items.get(name)
    
    def has(self, name: str) -> bool:
        """
        Check if item exists.
        
        Args:
            name: Item name
            
        Returns:
            True if item exists
        """
        return name in self.items
    
    def unregister(self, name: str) -> bool:
        """
        Unregister an item.
        
        Args:
            name: Item name
            
        Returns:
            True if item was unregistered
        """
        if name in self.items:
            del self.items[name]
            logger.debug(f"Unregistered {name} from {self.name}")
            return True
        return False
    
    def list_items(self) -> list[str]:
        """List all registered items."""
        return list(self.items.keys())
    
    def list_types(self) -> list[str]:
        """List all registered types."""
        return list(self.item_types.keys())
    
    def clear(self):
        """Clear all items."""
        self.items.clear()
        self.item_types.clear()
        logger.debug(f"Cleared {self.name}")
    
    def get_all(self) -> Dict[str, T]:
        """Get all items."""
        return self.items.copy()
    
    def count(self) -> int:
        """Get item count."""
        return len(self.items)


class FactoryRegistry(BaseRegistry[T]):
    """Registry with factory pattern support."""
    
    def __init__(self, name: str = "FactoryRegistry"):
        """
        Initialize factory registry.
        
        Args:
            name: Registry name
        """
        super().__init__(name)
        self.factories: Dict[str, Callable[..., T]] = {}
    
    def register_factory(self, name: str, factory: Callable[..., T]):
        """
        Register a factory function.
        
        Args:
            name: Factory name
            factory: Factory function
        """
        self.factories[name] = factory
        logger.debug(f"Registered factory {name} in {self.name}")
    
    def create_from_factory(self, name: str, *args, **kwargs) -> Optional[T]:
        """
        Create item from factory.
        
        Args:
            name: Factory name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Item instance or None
        """
        if name not in self.factories:
            logger.error(f"Factory {name} not found in {self.name}")
            return None
        
        factory = self.factories[name]
        item = factory(*args, **kwargs)
        self.items[name] = item
        return item




