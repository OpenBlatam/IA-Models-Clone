"""
Repository Base
==============

Base classes and utilities for repositories.
"""

from typing import Dict, Any, Optional, List, TypeVar, Generic
from abc import ABC, abstractmethod

from .base_models import BaseModel

T = TypeVar('T', bound=BaseModel)


class BaseRepository(ABC, Generic[T]):
    """Base repository with common operations."""
    
    @abstractmethod
    async def save(self, entity: T) -> None:
        """Save entity."""
        pass
    
    @abstractmethod
    async def get(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[T]:
        """Get all entities."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        pass
    
    async def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            True if exists
        """
        entity = await self.get(entity_id)
        return entity is not None
    
    async def count(self) -> int:
        """
        Get total count of entities.
        
        Returns:
            Total count
        """
        entities = await self.get_all()
        return len(entities)
    
    async def find_by(self, **criteria) -> List[T]:
        """
        Find entities by criteria.
        
        Args:
            **criteria: Search criteria
            
        Returns:
            List of matching entities
        """
        all_entities = await self.get_all()
        results = []
        
        for entity in all_entities:
            match = True
            for key, value in criteria.items():
                if hasattr(entity, key):
                    if getattr(entity, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                results.append(entity)
        
        return results
    
    async def update(self, entity_id: str, **updates) -> Optional[T]:
        """
        Update entity fields.
        
        Args:
            entity_id: Entity ID
            **updates: Fields to update
            
        Returns:
            Updated entity or None
        """
        entity = await self.get(entity_id)
        if entity:
            entity.update(**updates)
            await self.save(entity)
        return entity


class RepositoryMixin:
    """Mixin with common repository utilities."""
    
    def _serialize_entity(self, entity: BaseModel) -> Dict[str, Any]:
        """Serialize entity to dictionary."""
        return entity.to_dict()
    
    def _deserialize_entity(self, data: Dict[str, Any], entity_class: type[T]) -> T:
        """Deserialize dictionary to entity."""
        return entity_class.from_dict(data)
    
    def _filter_entities(
        self,
        entities: List[T],
        **criteria
    ) -> List[T]:
        """Filter entities by criteria."""
        results = []
        for entity in entities:
            match = True
            for key, value in criteria.items():
                if hasattr(entity, key):
                    if getattr(entity, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            if match:
                results.append(entity)
        return results




