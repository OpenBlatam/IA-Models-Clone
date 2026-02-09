"""
Repository Port
===============

Port for data persistence (Repository pattern).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel


class RepositoryPort(ABC):
    """Port for repository operations."""
    
    @abstractmethod
    async def create(self, entity: BaseModel) -> BaseModel:
        """Create entity."""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[BaseModel]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[BaseModel]:
        """Get all entities with optional filters."""
        pass
    
    @abstractmethod
    async def update(self, entity_id: str, entity: BaseModel) -> BaseModel:
        """Update entity."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete entity."""
        pass
    
    @abstractmethod
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists."""
        pass















