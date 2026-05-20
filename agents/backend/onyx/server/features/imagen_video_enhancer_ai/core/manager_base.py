"""
Manager Base
============

Base classes for managers.
"""

from typing import Dict, Any, Optional, List
from abc import ABC
import logging

logger = logging.getLogger(__name__)


class BaseManager(ABC):
    """Base class for managers with common functionality."""
    
    def __init__(self, name: str):
        """
        Initialize base manager.
        
        Args:
            name: Manager name
        """
        self.name = name
        self._initialized = False
        self._stats: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize manager."""
        if not self._initialized:
            await self._do_initialize()
            self._initialized = True
            logger.info(f"{self.name} initialized")
    
    async def _do_initialize(self) -> None:
        """Subclass-specific initialization."""
        pass
    
    async def shutdown(self) -> None:
        """Shutdown manager."""
        if self._initialized:
            await self._do_shutdown()
            self._initialized = False
            logger.info(f"{self.name} shutdown")
    
    async def _do_shutdown(self) -> None:
        """Subclass-specific shutdown."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get manager statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "name": self.name,
            "initialized": self._initialized,
            **self._stats
        }
    
    def update_stat(self, key: str, value: Any):
        """Update a statistic."""
        self._stats[key] = value
    
    def increment_stat(self, key: str, amount: int = 1):
        """Increment a statistic."""
        current = self._stats.get(key, 0)
        self._stats[key] = current + amount


class ManagerRegistry:
    """Registry for managing multiple managers."""
    
    def __init__(self):
        """Initialize manager registry."""
        self.managers: Dict[str, BaseManager] = {}
    
    def register(self, manager: BaseManager):
        """Register a manager."""
        self.managers[manager.name] = manager
    
    def get(self, name: str) -> Optional[BaseManager]:
        """Get manager by name."""
        return self.managers.get(name)
    
    async def initialize_all(self):
        """Initialize all managers."""
        for manager in self.managers.values():
            await manager.initialize()
    
    async def shutdown_all(self):
        """Shutdown all managers."""
        for manager in self.managers.values():
            await manager.shutdown()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all managers."""
        return {name: manager.get_stats() for name, manager in self.managers.items()}




