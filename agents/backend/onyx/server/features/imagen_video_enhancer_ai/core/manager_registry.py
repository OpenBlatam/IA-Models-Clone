"""
Manager Registry
================

Consolidated registry for all manager types.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Generic
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseManager(ABC):
    """Base manager interface."""
    
    @abstractmethod
    def initialize(self):
        """Initialize manager."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Shutdown manager."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        pass


class ManagerRegistry:
    """Registry for all managers."""
    
    def __init__(self):
        """Initialize manager registry."""
        self.managers: Dict[str, BaseManager] = {}
        self.manager_types: Dict[str, Type[BaseManager]] = {}
    
    def register_type(self, name: str, manager_type: Type[BaseManager]):
        """
        Register a manager type.
        
        Args:
            name: Manager name
            manager_type: Manager class
        """
        self.manager_types[name] = manager_type
        logger.debug(f"Registered manager type: {name}")
    
    def create(self, name: str, *args, **kwargs) -> Optional[BaseManager]:
        """
        Create a manager instance.
        
        Args:
            name: Manager name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Manager instance or None
        """
        if name not in self.manager_types:
            logger.error(f"Manager type {name} not found")
            return None
        
        manager_type = self.manager_types[name]
        manager = manager_type(*args, **kwargs)
        self.managers[name] = manager
        return manager
    
    def register(self, name: str, manager: BaseManager):
        """
        Register a manager instance.
        
        Args:
            name: Manager name
            manager: Manager instance
        """
        self.managers[name] = manager
        logger.debug(f"Registered manager: {name}")
    
    def get(self, name: str) -> Optional[BaseManager]:
        """
        Get manager by name.
        
        Args:
            name: Manager name
            
        Returns:
            Manager instance or None
        """
        return self.managers.get(name)
    
    def initialize_all(self):
        """Initialize all managers."""
        for name, manager in self.managers.items():
            try:
                manager.initialize()
                logger.info(f"Initialized manager: {name}")
            except Exception as e:
                logger.error(f"Error initializing manager {name}: {e}")
    
    def shutdown_all(self):
        """Shutdown all managers."""
        for name, manager in self.managers.items():
            try:
                manager.shutdown()
                logger.info(f"Shutdown manager: {name}")
            except Exception as e:
                logger.error(f"Error shutting down manager {name}: {e}")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all managers."""
        stats = {}
        for name, manager in self.managers.items():
            try:
                stats[name] = manager.get_stats()
            except Exception as e:
                logger.error(f"Error getting stats from manager {name}: {e}")
                stats[name] = {"error": str(e)}
        return stats
    
    def list_managers(self) -> list[str]:
        """List all registered managers."""
        return list(self.managers.keys())




