"""
Factory Base
============

Base factory pattern for object creation.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Generic, Callable, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseFactory(ABC, Generic[T]):
    """Base factory interface."""
    
    def __init__(self, name: str = "Factory"):
        """
        Initialize factory.
        
        Args:
            name: Factory name
        """
        self.name = name
        self.creators: Dict[str, Callable[..., T]] = {}
        self.types: Dict[str, Type[T]] = {}
    
    def register_type(self, name: str, item_type: Type[T]):
        """
        Register a type.
        
        Args:
            name: Type name
            item_type: Type class
        """
        self.types[name] = item_type
        logger.debug(f"Registered type {name} in {self.name}")
    
    def register_creator(self, name: str, creator: Callable[..., T]):
        """
        Register a creator function.
        
        Args:
            name: Creator name
            creator: Creator function
        """
        self.creators[name] = creator
        logger.debug(f"Registered creator {name} in {self.name}")
    
    def create(self, name: str, *args, **kwargs) -> Optional[T]:
        """
        Create instance.
        
        Args:
            name: Type or creator name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Instance or None
        """
        # Try creator first
        if name in self.creators:
            creator = self.creators[name]
            return creator(*args, **kwargs)
        
        # Try type
        if name in self.types:
            item_type = self.types[name]
            return item_type(*args, **kwargs)
        
        logger.error(f"Unknown type/creator: {name} in {self.name}")
        return None
    
    def has(self, name: str) -> bool:
        """
        Check if type/creator exists.
        
        Args:
            name: Type or creator name
            
        Returns:
            True if exists
        """
        return name in self.creators or name in self.types
    
    def list_types(self) -> List[str]:
        """List all registered types."""
        return list(self.types.keys())
    
    def list_creators(self) -> List[str]:
        """List all registered creators."""
        return list(self.creators.keys())


class BuilderFactory(BaseFactory[T]):
    """Factory with builder pattern support."""
    
    def __init__(self, name: str = "BuilderFactory"):
        """
        Initialize builder factory.
        
        Args:
            name: Factory name
        """
        super().__init__(name)
        self.builders: Dict[str, Callable[[], Any]] = {}
    
    def register_builder(self, name: str, builder: Callable[[], Any]):
        """
        Register a builder.
        
        Args:
            name: Builder name
            builder: Builder function that returns a builder instance
        """
        self.builders[name] = builder
        logger.debug(f"Registered builder {name} in {self.name}")
    
    def build(self, name: str, *args, **kwargs) -> Optional[T]:
        """
        Build instance using builder.
        
        Args:
            name: Builder name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Built instance or None
        """
        if name not in self.builders:
            logger.error(f"Builder {name} not found in {self.name}")
            return None
        
        builder_func = self.builders[name]
        builder = builder_func()
        
        # Apply arguments if builder supports it
        if hasattr(builder, 'build'):
            return builder.build()
        else:
            return builder

